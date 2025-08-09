import os
import json
import time
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from openai import AzureOpenAI, RateLimitError
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


def process_gpt5_single_question(client, question_data, max_retries=3):
    """GPT-5 단일 질문 처리 (ChatCompletion API 사용)"""
    retries = 0
    while retries <= max_retries:
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a multiple-choice answerer. Read the question and options, choose the single best option, and output exactly one uppercase letter (A, B, C, D, or E)."
                    },
                    {
                        "role": "user", 
                        "content": question_data["question"]
                    }
                ],
                model="gpt-5"
                # formatting_reenabled 파라미터 제거 (지원되지 않을 수 있음)
            )
            
            raw_response = response.choices[0].message.content
            
            # MultipleChoicesFiveParser와 유사한 처리
            parser = MultipleChoicesFiveParser()
            parsed_result = parser.parse(raw_response)
            
            return {
                "id": question_data["id"],
                "trial": 0,
                "answer": question_data["answer"],
                "pred": parsed_result[0] if parsed_result else None,
                "response": parsed_result[1] if parsed_result and len(parsed_result) > 1 else raw_response,
            }
            
        except openai.APIError as e:
            if e.status_code == 503:
                # 503 에러의 경우 더 긴 대기 시간
                delay = (retries + 1) * 60  # 1분, 2분, 3분
                logger.warning(f"Service temporarily unavailable for {question_data['id']}. Retrying in {delay} seconds...")
                time.sleep(delay)
                retries += 1
            else:
                logger.error(f"API Error processing {question_data['id']}: {e}")
                retries += 1
        except RateLimitError:
            retries += 1
            delay = retries * 30
            logger.warning(f"Rate limit hit for {question_data['id']}. Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Error processing {question_data['id']}: {e}")
            retries += 1
            
    logger.error(f"Failed to process {question_data['id']} after {max_retries} retries")
    return None


def process_gpt5_batch_parallel(client, batch_data, max_workers=3, max_retries=3):
    """GPT-5 배치를 병렬로 처리 (동시 요청 수 줄임)"""
    results = []
    
    # 503 에러가 많이 발생하므로 동시 요청 수를 줄임
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 질문을 병렬로 제출
        future_to_question = {
            executor.submit(process_gpt5_single_question, client, question_data, max_retries): question_data
            for question_data in batch_data
        }
        
        # 결과 수집
        for future in as_completed(future_to_question):
            question_data = future_to_question[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                logger.error(f"Question {question_data['id']} generated an exception: {exc}")
            
            # 각 요청 사이에 약간의 지연 추가
            time.sleep(0.1)
    
    return results


def benchmark(args):
    is_debug = args.is_debug
    max_retries = args.max_retries
    delay_increment = 30

    num_debug_samples = args.num_debug_samples
    batch_size = args.batch_size
    max_tokens = args.max_tokens
    temperature = args.temperature
    
    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    model_version = os.getenv("OPENAI_MODEL_VERSION", "2025-08-08")
    
    # GPT-5인지 확인
    is_gpt5 = model_name in ["gpt-5", "gpt-5-chat"]
    
    if is_gpt5:
        # GPT-5용 AzureOpenAI 클라이언트 설정
        client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        
        logger.info(f"Using GPT-5 direct API for model: {model_name}")
    else:
        # 기존 LangChain 체인 사용
        llm, model_name = get_llm_client(
            args.model_provider, args.hf_model_id, temperature, max_tokens, max_retries
        )
        prompt_template = get_prompt_template(args.template_type)
        chain = prompt_template | llm | MultipleChoicesFiveParser()
        logger.info(f"Using LangChain for model: {model_name}")

    click_ds = load_dataset("EunsuKim/CLIcK")["train"]

    if is_debug:
        click_ds = click_ds.select(range(num_debug_samples))

    all_batch = [
        {"id": x["id"], "question": get_prompt(x), "answer": get_answer(x)}
        for x in tqdm(click_ds)
    ]
    responses = []

    logger.info(f"====== [START] Generate answers to questions given by LLM. =====")
    logger.info(f"====== deployment name: {model_name}, model version: {model_version} =====")
    t0 = time.time()

    if is_gpt5:
        # GPT-5 병렬 처리 (더 보수적인 설정)
        with tqdm(total=len(all_batch), desc="Processing Answers (GPT-5)") as pbar:
            for i in range(0, len(all_batch), batch_size):
                mini_batch = all_batch[i : i + batch_size]
                
                # 503 에러를 줄이기 위해 더 적은 동시 요청
                max_workers = min(batch_size, 2)  # 최대 2개의 동시 요청
                batch_results = process_gpt5_batch_parallel(client, mini_batch, max_workers, max_retries)
                responses.extend(batch_results)
                
                pbar.set_postfix({
                    "current_batch": f"{i//batch_size + 1}/{(len(all_batch) + (batch_size-1))//batch_size}",
                    "success_rate": f"{len(batch_results)}/{len(mini_batch)}"
                })
                pbar.update(len(mini_batch))
                
                # 배치 간 휴식 시간 추가
                if i + batch_size < len(all_batch):
                    time.sleep(1)
    else:
        # 기존 LangChain 처리 (변경 없음)
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
                                f"Max retries reached this batch. Adding failed responses for this batch."
                            )
                            # 실패한 질문들에 대해 기본값으로 추가
                            for qna in mini_batch:
                                responses.append(
                                    {
                                        "id": qna["id"],
                                        "trial": 0,
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
                                    "id": qna["id"],
                                    "trial": 0,
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
                                    "id": qna["id"],
                                    "trial": 0,
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
    csv_path = f"results/[CLIcK] {model_name}-{model_version}.csv"
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

    # 세부 카테고리별 평균
    category_avg = (
        result.groupby(["category_big", "category"])
        .agg(correct_mean=("correct", "mean"), correct_count=("correct", "size"))
        .reset_index()
    )
    print(category_avg)
    
    # 대분류 카테고리별 평균
    category_big_avg = (
        result.groupby(["category_big"])
        .agg(correct_mean=("correct", "mean"), correct_count=("correct", "size"))
        .reset_index()
    )
    print(category_big_avg)
    
    overall_avg = category_avg["correct_mean"].mean()
    print(f"Overall Average: {overall_avg}")

    os.makedirs("evals", exist_ok=True)
    filename = csv_path.split("/")[-1].split(".")[0]
    category_avg.to_csv(f"evals/{filename}-eval.csv", index=False)
    category_big_avg.to_csv(f"evals/{filename}-eval-avg.csv", index=False)


if __name__ == "__main__":
    dotenv_path = os.getenv('DOTENV_PATH', '.env')
    load_dotenv(dotenv_path)
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument("--is_debug", type=str2bool, default=True)
    parser.add_argument("--num_debug_samples", type=int, default=20)
    parser.add_argument("--model_provider", type=str, default="azureopenai")
    parser.add_argument(
        "--hf_model_id", type=str, default="microsoft/Phi-3.5-MoE-instruct"
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
