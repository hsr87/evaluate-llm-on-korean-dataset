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

from prompts import TYPE_1, TYPE_2, TYPE_3, TYPE_4, TYPE_MMLU_FEW_SHOT
from util.common_helper import (
    str2bool,
    format_timespan,
    get_prompt_template,
    get_llm_client,
)
from logger import logger
from util.custom_parser import MultipleChoicesFourParser


def generate_few_shots_prompt(data):
    prompt = ""
    for i, row in enumerate(data):
        # prompt += f"## Example {i+1}:\n"
        prompt += f"질문 (Question): {row['question']}\n"
        prompt += f"보기 (Options)\nA: {row['A']}, B: {row['B']}, C: {row['C']}, D: {row['D']}\n"
        prompt += f"정답 (Answer): {row['answer']}\n\n"
    return prompt


def get_prompt(x, few_shots=None) -> str:
    if few_shots is None:
        return TYPE_2.format(
            QUESTION=x["question"],
            A=x["A"],
            B=x["B"],
            C=x["C"],
            D=x["D"],
        )
    else:
        return TYPE_MMLU_FEW_SHOT.format(
            FEW_SHOTS=few_shots,
            QUESTION=x["question"],
            A=x["A"],
            B=x["B"],
            C=x["C"],
            D=x["D"],
        )


def get_answer(x) -> str:
    return x["answer"].upper().strip()


def map_answer(answer):
    answer_mapping = {1: "A", 2: "B", 3: "C", 4: "D"}
    return answer_mapping[answer]


def convert_to_pascal_case(category):
    return "-".join(word.capitalize() for word in category.split("_"))


def benchmark(args):

    is_debug = args.is_debug
    max_retries = args.max_retries
    delay_increment = 30
    few_shots_label = "5shot" if args.use_few_shots else "0shot"

    num_debug_samples = args.num_debug_samples
    batch_size = args.batch_size
    max_tokens = args.max_tokens
    temperature = args.temperature

    if args.is_hard:
        hf_dataset_id = "HAERAE-HUB/KMMLU-HARD"
        dataset_name = "KMMLU-HARD"
        kmmlu_category = [
            "accounting",
            "agricultural_sciences",
            "aviation_engineering_and_maintenance",
            "biology",
            "chemical_engineering",
            "chemistry",
            "civil_engineering",
            "computer_science",
            "construction",
            "criminal_law",
            "ecology",
            "economics",
            "education",
            "electrical_engineering",
            "electronics_engineering",
            "energy_management",
            "environmental_science",
            "fashion",
            "food_processing",
            "gas_technology_and_engineering",
            "geomatics",
            "health",
            "industrial_engineer",
            "information_technology",
            "interior_architecture_and_design",
            "korean_history",
            "law",
            "machine_design_and_manufacturing",
            "management",
            "maritime_engineering",
            "marketing",
            "materials_engineering",
            "math",
            "mechanical_engineering",
            "nondestructive_testing",
            "patent",
            "political_science_and_sociology",
            "psychology",
            "public_safety",
            "railway_and_automotive_engineering",
            "real_estate",
            "refrigerating_machinery",
            "social_welfare",
            "taxation",
            "telecommunications_and_wireless_technology",
        ]

    else:
        hf_dataset_id = "HAERAE-HUB/KMMLU"
        dataset_name = "KMMLU"
        kmmlu_category = [
            "Accounting",
            "Agricultural-Sciences",
            "Aviation-Engineering-and-Maintenance",
            "Biology",
            "Chemical-Engineering",
            "Chemistry",
            "Civil-Engineering",
            "Computer-Science",
            "Construction",
            "Criminal-Law",
            "Ecology",
            "Economics",
            "Education",
            "Electrical-Engineering",
            "Electronics-Engineering",
            "Energy-Management",
            "Environmental-Science",
            "Fashion",
            "Food-Processing",
            "Gas-Technology-and-Engineering",
            "Geomatics",
            "Health",
            "Industrial-Engineer",
            "Information-Technology",
            "Interior-Architecture-and-Design",
            "Korean-History",
            "Law",
            "Machine-Design-and-Manufacturing",
            "Management",
            "Maritime-Engineering",
            "Marketing",
            "Materials-Engineering",
            "Math",
            "Mechanical-Engineering",
            "Nondestructive-Testing",
            "Patent",
            "Political-Science-and-Sociology",
            "Psychology",
            "Public-Safety",
            "Railway-and-Automotive-Engineering",
            "Real-Estate",
            "Refrigerating-Machinery",
            "Social-Welfare",
            "Taxation",
            "Telecommunications-and-Wireless-Technology",
        ]

    llm, model_name = get_llm_client(
        args.model_provider, args.hf_model_id, temperature, max_tokens, max_retries
    )
    model_version = (
        os.getenv("OPENAI_MODEL_VERSION")
        if args.model_provider == "azureopenai"
        else None
    )

    logger.info(f"====== [START] Generate answers to questions given by LLM. =====")
    if args.use_few_shots:
        logger.info(f"===== Use Few-shots Prompt.")
    else:
        logger.info(f"===== Use Zero-shot Prompt.")
    logger.info(
        f"====== deployment name: {model_name}, model version: {model_version} ====="
    )
    responses = []

    # Load the datasets and append to the list with their respective categories
    failed_categories = []
    successful_categories = []
    
    # 문제가 되는 카테고리들을 임시로 건너뛰기
    problematic_categories = []  # 필요시 추가
    
    for c in kmmlu_category:
        if c in problematic_categories:
            logger.warning(f"##### Category [{c}] SKIPPED (in problematic_categories list)")
            continue
            
        try:
            logger.info(f"##### Category [{c}] Processing...")
            ds_dict = load_dataset(hf_dataset_id, c)

            # For few-shot prompts, we need to generate a prompt with examples
            ds_dev = ds_dict["dev"]
            ds_dev = ds_dev.map(lambda x: {"answer": map_answer(x["answer"])})
            if args.is_hard:
                ds_dev = ds_dev.map(
                    lambda x: {"category": convert_to_pascal_case(x["category"])}
                )
            else:
                ds_dev = ds_dev.rename_column("Category", "category")

            ds = ds_dict["test"]
            ds = ds.map(lambda x: {"answer": map_answer(x["answer"])})
            if args.is_hard:
                ds = ds.map(lambda x: {"category": convert_to_pascal_case(x["category"])})
            else:
                ds = ds.rename_column("Category", "category")

            if is_debug:
                ds = ds.select(range(num_debug_samples))

            if args.use_few_shots:
                few_shots_prompt = generate_few_shots_prompt(ds_dev)
            else:
                few_shots_prompt = None

            all_batch = [
                {
                    "category": c,
                    "question": get_prompt(x, few_shots_prompt),
                    "answer": get_answer(x),
                }
                for x in tqdm(ds, desc=f"Preparing {c}")
            ]

            prompt_template = get_prompt_template(args.template_type)
            chain = prompt_template | llm | MultipleChoicesFourParser()

            t0 = time.time()
            category_responses = []
            category_success = True

            with tqdm(total=len(all_batch), desc=f"Processing {c}") as pbar:
                for i in range(0, len(all_batch), batch_size):
                    mini_batch = all_batch[i : i + batch_size]
                    retries = 0
                    batch_success = False

                    while retries <= max_retries:
                        try:
                            preds = chain.batch(mini_batch, {"max_concurrency": batch_size})
                            # If no exception, add questions and answers to all_answers
                            for qna, pred in zip(mini_batch, preds):
                                category_responses.append(
                                    {
                                        "category": qna["category"],
                                        "answer": qna["answer"],
                                        "pred": pred[0],
                                        "response": pred[1],
                                    }
                                )
                            batch_success = True
                            break  # Exit the retry loop once successful
                        except RateLimitError as rate_limit_error:
                            delay = (retries + 1) * delay_increment
                            logger.warning(
                                f"[{c}] {rate_limit_error}. Retrying in {delay} seconds..."
                            )
                            time.sleep(delay)
                            retries += 1

                            if retries > max_retries:
                                logger.error(
                                    f"[{c}] Max retries reached this batch. Adding failed responses."
                                )
                                # 실패한 질문들에 대해 기본값으로 추가
                                for qna in mini_batch:
                                    category_responses.append(
                                        {
                                            "category": qna["category"],
                                            "answer": qna["answer"],
                                            "pred": "FAILED",
                                            "response": "RATE_LIMIT_ERROR",
                                        }
                                    )
                                batch_success = True  # 실패해도 계속 진행
                                break
                        except openai.BadRequestError as e:
                            logger.error(f"[{c}] BadRequestError: {e}. Adding failed responses for this batch.")
                            logger.info(f"[{c}] Question sample: {mini_batch[0]['question'][:100]}...")
                            # 실패한 질문들에 대해 기본값으로 추가
                            for qna in mini_batch:
                                category_responses.append(
                                    {
                                        "category": qna["category"],
                                        "answer": qna["answer"],
                                        "pred": "FAILED",
                                        "response": "BAD_REQUEST_ERROR",
                                    }
                                )
                            batch_success = True  # 실패해도 계속 진행
                            break
                        except Exception as e:
                            logger.error(f"[{c}] Error in process_inputs: {e}. Adding failed responses for this batch.")
                            # 실패한 질문들에 대해 기본값으로 추가
                            for qna in mini_batch:
                                category_responses.append(
                                    {
                                        "category": qna["category"],
                                        "answer": qna["answer"],
                                        "pred": "FAILED",
                                        "response": f"ERROR: {str(e)}",
                                    }
                                )
                            batch_success = True  # 실패해도 계속 진행
                            break

                    if not batch_success:
                        logger.error(f"[{c}] Failed to process batch {i//batch_size + 1}, but continuing...")

                    pbar.set_postfix(
                        {
                            "current_batch": f"{i//batch_size + 1}/{(len(all_batch) + (batch_size-1))//batch_size}",
                            "success": "✓" if batch_success else "✗"
                        }
                    )
                    pbar.update(len(mini_batch))

            # 카테고리별 결과를 전체 응답에 추가
            responses.extend(category_responses)
            
            t1 = time.time()
            acc = evaluate_each_category(category_responses, c)
            timespan = format_timespan(t1 - t0)
            logger.info(f"##### Category [{c}] accuracy: {acc}% ({len(category_responses)} responses)")
            logger.info(f"##### Generating Answers for Category [{c}] took {timespan}")
            successful_categories.append(c)
            
        except Exception as category_error:
            logger.error(f"##### CRITICAL ERROR in Category [{c}]: {category_error}")
            logger.error(f"##### Skipping category [{c}] and continuing with next category...")
            failed_categories.append(c)
            continue

    # 최종 요약
    logger.info(f"====== PROCESSING SUMMARY =====")
    logger.info(f"Successfully processed: {len(successful_categories)}/{len(kmmlu_category)} categories")
    logger.info(f"Successful categories: {successful_categories}")
    if failed_categories:
        logger.warning(f"Failed categories: {failed_categories}")
    
    logger.info(
        "====== [DONE] Completed Generating Answers to Questions given by LLM. ====="
    )
    
    if not responses:
        logger.error("No successful responses were generated. Skipping evaluation.")
        return

    df = pd.DataFrame(responses)
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/[{dataset_name}] {model_name}-{model_version}-{few_shots_label}.csv"
    logger.info(f"====== Generated CSV file - CSV_PATH: {csv_path} =====")
    df.to_csv(csv_path, index=False)

    logger.info(f"====== [START] Evaluation start - CSV_PATH: {csv_path} =====")
    evaluate(csv_path)
    logger.info(f"====== [START] Evaluation end =====")


def evaluate_each_category(responses, category):
    df = pd.DataFrame(responses)
    
    # Check if DataFrame is empty or missing required columns
    if df.empty or "category" not in df.columns:
        logger.warning(f"No valid responses found for category {category}. Returning 0% accuracy.")
        return 0.0
    
    df = df[df["category"] == category]
    
    # Check if any rows match the category
    if df.empty:
        logger.warning(f"No responses found for category {category}. Returning 0% accuracy.")
        return 0.0
    
    # FAILED 응답 제외
    original_count = len(df)
    failed_count = len(df[df["pred"] == "FAILED"])
    if failed_count > 0:
        logger.info(f"Category [{category}]: Excluding {failed_count} FAILED responses from {original_count} total")
        df = df[df["pred"] != "FAILED"]
    
    if df.empty:
        logger.warning(f"No valid responses found for category {category} after filtering. Returning 0% accuracy.")
        return 0.0
    
    df["correct"] = df["answer"] == df["pred"]
    acc = round(df["correct"].mean() * 100, 2)
    logger.info(f"Category [{category}]: {len(df)} valid responses, accuracy: {acc}%")
    return acc


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
    load_dotenv(dotenv_path)
    parser = argparse.ArgumentParser(description="Options")

    parser.add_argument("--is_debug", type=str2bool, default=False)
    parser.add_argument("--num_debug_samples", type=int, default=10)
    parser.add_argument("--model_provider", type=str, default="azureopenai")
    parser.add_argument(
        "--hf_model_id", type=str, default="microsoft/Phi-3.5-mini-instruct"
    )
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--template_type", type=str, default="chat")
    parser.add_argument("--is_hard", type=str2bool, default=True)
    parser.add_argument("--use_few_shots", type=str2bool, default=True)

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
