import os
import json
import time
import argparse
import openai
from openai import RateLimitError
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
from prompts import TYPE_1, TYPE_2, TYPE_3, TYPE_4
from util.custom_parser import MultipleChoicesFiveParser

from util.common_helper import (
    str2bool,
    format_timespan,
    get_prompt_template,
    get_llm_client,
)

from logger import logger


def get_prompt(x) -> str:
    return TYPE_4.format(
        QUESTION=x["question"],
        A=x["a"],
        B=x["b"],
        C=x["c"],
        D=x["d"],
        E=x["e"],
    )


def get_answer(x) -> str:
    return x["answer"].upper().strip()


def benchmark(args):

    is_debug = args.is_debug
    max_retries = args.max_retries
    delay_increment = 30

    num_debug_samples = args.num_debug_samples
    batch_size = args.batch_size
    max_tokens = args.max_tokens
    temperature = args.temperature
    llm, model_name = get_llm_client(
        args.model_provider, args.hf_model_id, temperature, max_tokens, max_retries
    )
    model_version = (
        os.getenv("OPENAI_MODEL_VERSION")
        if args.model_provider == "azureopenai"
        else None
    )

    # Initialize an empty list to store the datasets
    haerae_ds_list = []
    haerae_category = [
        "General Knowledge",
        "History",
        "Loan Words",
        "Rare Words",
        "Reading Comprehension",
        "Standard Nomenclature",
    ]

    # Load the datasets and append to the list with their respective categories
    for c in haerae_category:
        ds = load_dataset("HAERAE-HUB/HAE_RAE_BENCH_1.0", c)["test"]
        df = ds.to_pandas()
        df["category"] = c
        haerae_ds_list.append(df)

    # Concatenate all the dataframes into a single dataframe
    combined_df = pd.concat(haerae_ds_list, ignore_index=True)
    haerae_ds = Dataset.from_pandas(combined_df)

    if is_debug:
        haerae_ds = haerae_ds.select(range(num_debug_samples))

    all_batch = [
        {"category": x["category"], "question": get_prompt(x), "answer": get_answer(x)}
        for x in tqdm(haerae_ds)
    ]
    responses = []
    prompt_template = get_prompt_template(args.template_type)
    chain = prompt_template | llm | MultipleChoicesFiveParser()

    logger.info(f"====== [START] Generate answers to questions given by LLM. =====")
    logger.info(
        f"====== deployment name: {model_name}, model version: {model_version} ====="
    )
    t0 = time.time()

    with tqdm(total=len(all_batch), desc="Processing Answers") as pbar:

        for i in range(0, len(all_batch), batch_size):
            mini_batch = all_batch[i : i + batch_size]
            retries = 0

            while retries <= max_retries:
                try:
                    preds = chain.batch(mini_batch, {"max_concurrency": batch_size})
                    # If no exception, add questions and answers to all_answers
                    for qna, pred in zip(mini_batch, preds):
                        responses.append(
                            {
                                "category": qna["category"],
                                "answer": qna["answer"],
                                "pred": pred[0],
                                "response": pred[1],
                            }
                        )
                    break  # Exit the retry loop once successful
                except RateLimitError as rate_limit_error:
                    delay = (retries + 1) * delay_increment
                    logger.warning(
                        f"{rate_limit_error}. Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
                    retries += 1

                    if retries > max_retries:
                        logger.error(
                            f"Max retries reached this batch. Adding failed responses for this batch."
                        )
                        # 실패한 질문들에 대해 기본값으로 추가
                        for qna in mini_batch:
                            responses.append(
                                {
                                    "category": qna["category"],
                                    "answer": qna["answer"],
                                    "pred": "FAILED",
                                    "response": "RATE_LIMIT_ERROR",
                                }
                            )
                        break
                except openai.BadRequestError as e:
                    logger.error(f"BadRequestError: {e}. Adding failed responses for this batch.")
                    logger.info(f"Question sample: {mini_batch[0]['question'][:100]}...")
                    # 실패한 질문들에 대해 기본값으로 추가
                    for qna in mini_batch:
                        responses.append(
                            {
                                "category": qna["category"],
                                "answer": qna["answer"],
                                "pred": "FAILED",
                                "response": "BAD_REQUEST_ERROR",
                            }
                        )
                    break
                except Exception as e:
                    logger.error(f"Error in process_inputs: {e}. Adding failed responses for this batch.")
                    # 실패한 질문들에 대해 기본값으로 추가
                    for qna in mini_batch:
                        responses.append(
                            {
                                "category": qna["category"],
                                "answer": qna["answer"],
                                "pred": "FAILED",
                                "response": f"ERROR: {str(e)}",
                            }
                        )
                    break

            pbar.set_postfix(
                {
                    "current_batch": f"{i//batch_size + 1}/{(len(all_batch) + (batch_size-1))//batch_size}"
                }
            )
            pbar.update(len(mini_batch))

    t1 = time.time()
    timespan = format_timespan(t1 - t0)
    logger.info(f"===== [DONE] Generating Answer dataset took {timespan}")

    if not responses:
        logger.error("No successful responses were generated. Skipping evaluation.")
        return

    df = pd.DataFrame(responses)
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/[HAERAE] {model_name}-{model_version}.csv"
    logger.info(f"====== Generated CSV file - CSV_PATH: {csv_path} =====")
    df.to_csv(csv_path, index=False)

    logger.info(f"====== [START] Evaluation start - CSV_PATH: {csv_path} =====")
    evaluate(csv_path)
    logger.info(f"====== [START] Evaluation end =====")


def evaluate(csv_path):
    # Check if file exists and has content
    if not os.path.exists(csv_path):
        logger.error(f"CSV file does not exist: {csv_path}")
        return
    
    if os.path.getsize(csv_path) == 0:
        logger.error(f"CSV file is empty: {csv_path}")
        return
    
    try:
        result = pd.read_csv(csv_path)
        if result.empty:
            logger.error(f"CSV file contains no data: {csv_path}")
            return
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file has no columns to parse: {csv_path}")
        return
    
    # FAILED 응답 필터링 및 로깅
    original_count = len(result)
    failed_count = len(result[result["pred"] == "FAILED"])
    if failed_count > 0:
        logger.warning(f"Found {failed_count} FAILED responses out of {original_count} total responses")
        logger.info(f"Excluding FAILED responses from accuracy calculation")
        result = result[result["pred"] != "FAILED"]
        logger.info(f"Evaluating on {len(result)} valid responses")
    
    result["correct"] = result["answer"] == result["pred"]

    category_avg = (
        result.groupby(["category"])
        .agg(correct_mean=("correct", "mean"), correct_count=("correct", "size"))
        .reset_index()
    )
    print(category_avg)
    overall_avg = category_avg["correct_mean"].mean()
    print(f"Overall Average: {overall_avg}")

    os.makedirs("evals", exist_ok=True)
    filename = csv_path.split("/")[-1].split(".")[0]
    category_avg.to_csv(f"evals/{filename}-eval.csv", index=False)


if __name__ == "__main__":
    dotenv_path = os.getenv('DOTENV_PATH', '.env')
    load_dotenv(dotenv_path, override=True)
    parser = argparse.ArgumentParser(description="Options")

    parser.add_argument("--is_debug", type=str2bool, default=True)
    parser.add_argument("--num_debug_samples", type=int, default=20)
    parser.add_argument("--model_provider", type=str, default="azureopenai")
    parser.add_argument(
        "--hf_model_id", type=str, default="microsoft/Phi-3.5-mini-instruct"
    )
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--template_type", type=str, default="chat")

    args = parser.parse_args()
    valid_providers = ["azureopenai", "openai", "azureml", "azureai", "huggingface"]
    assert (
        args.model_provider in valid_providers
    ), f"Invalid 'model_provider' value. Please choose from {valid_providers}."

    valid_template_types = ["basic", "chat"]
    assert (
        args.template_type in valid_template_types
    ), f"Invalid 'template_type' value. Please choose from {valid_template_types}."

    logger.info(args)
    benchmark(args)
