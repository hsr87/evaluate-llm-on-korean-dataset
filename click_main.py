import os
import json
import time
import argparse

import openai
from openai import RateLimitError
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset

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
    num_choices = len(x["choices"])
    if num_choices == 4:
        if x["paragraph"] != "":  # Use Type 1 Prompt
            return TYPE_1.format(
                CONTEXT=x["paragraph"],
                QUESTION=x["question"],
                A=x["choices"][0],
                B=x["choices"][1],
                C=x["choices"][2],
                D=x["choices"][3],
            )
        else:
            return TYPE_2.format(
                QUESTION=x["question"],
                A=x["choices"][0],
                B=x["choices"][1],
                C=x["choices"][2],
                D=x["choices"][3],
            )
    elif num_choices == 5:
        if x["paragraph"] != "":
            return TYPE_3.format(
                CONTEXT=x["paragraph"],
                QUESTION=x["question"],
                A=x["choices"][0],
                B=x["choices"][1],
                C=x["choices"][2],
                D=x["choices"][3],
                E=x["choices"][4],
            )
        else:
            return TYPE_4.format(
                QUESTION=x["question"],
                A=x["choices"][0],
                B=x["choices"][1],
                C=x["choices"][2],
                D=x["choices"][3],
                E=x["choices"][4],
            )
    else:
        raise ValueError(f"Invalid number of choices: {num_choices} (ID: {x['id']})")


def get_answer(x) -> str:
    answer_idx = [xx.strip() for xx in x["choices"]].index(x["answer"].strip())
    if answer_idx == -1:
        raise ValueError(f"Answer not found in choices: {x['answer']} (ID: {x['id']})")
    return chr(0x41 + answer_idx)  # answer_idx = 0 -> answer = "A"


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

    click_ds = load_dataset("EunsuKim/CLIcK")["train"]

    if is_debug:
        click_ds = click_ds.select(range(num_debug_samples))

    all_batch = [
        {"id": x["id"], "question": get_prompt(x), "answer": get_answer(x)}
        for x in tqdm(click_ds)
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
                                "id": qna["id"],
                                "trial": 0,
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
                            f"Max retries reached this batch. Skipping to next batch."
                        )
                        break
                except openai.BadRequestError as e:
                    logger.error(f"BadRequestError: {e}. Skipping this batch.")
                    logger.info(f"Question: {qna['question']}")
                    break
                except Exception as e:
                    logger.error(f"Error in process_inputs: {e}")
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

    df = pd.DataFrame(responses)
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/[CLIcK] {model_name}-{model_version}.csv"
    logger.info(f"====== Generated CSV file - CSV_PATH: {csv_path} =====")
    df.to_csv(csv_path, index=False)

    logger.info(f"====== [START] Evaluation start - CSV_PATH: {csv_path} =====")
    evaluate(csv_path)
    logger.info(f"====== [START] Evaluation end =====")


def evaluate(csv_path):
    result = pd.read_csv(csv_path)
    with open("id_to_category.json", "r") as json_file:
        id_to_category = json.load(json_file)

    result["category"] = result["id"].map(id_to_category)
    result["correct"] = result["answer"] == result["pred"]
    result["category_big"] = result["category"].apply(
        lambda x: (
            "Culture"
            if x
            in [
                "Economy",
                "Geography",
                "History",
                "Law",
                "Politics",
                "Popular",
                "Society",
                "Tradition",
                "Pop Culture",
            ]
            else ("Language" if x in ["Functional", "Textual", "Grammar"] else "Other")
        )
    )

    category_avg = (
        result.groupby(["category_big", "category"])
        .agg(correct_mean=("correct", "mean"), correct_count=("correct", "size"))
        .reset_index()
    )
    print(category_avg)

    category_big_avg = (
        result.groupby("category_big")
        .agg(correct_mean=("correct", "mean"), correct_count=("correct", "size"))
        .reset_index()
    )
    print(category_big_avg)

    os.makedirs("evals", exist_ok=True)
    filename = csv_path.split("/")[-1].split(".")[0]
    category_avg.to_csv(f"evals/{filename}-eval.csv", index=False)
    category_big_avg.to_csv(f"evals/{filename}-eval-avg.csv", index=False)


if __name__ == "__main__":
    load_dotenv(".env")
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument("--is_debug", type=str2bool, default=True)
    parser.add_argument("--num_debug_samples", type=int, default=20)
    parser.add_argument("--model_provider", type=str, default="azureopenai")
    parser.add_argument(
        "--hf_model_id", type=str, default="microsoft/Phi-3.5-MoE-instruct"
    )
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--template_type", type=str, default="basic")

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
