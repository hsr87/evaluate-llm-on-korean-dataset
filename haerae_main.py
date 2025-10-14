import os
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

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


def load_existing_results(csv_path):
    """기존 결과 파일이 있으면 로드"""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                logger.info(f"Found existing results: {len(df)} records in {csv_path}")
                return df
        except Exception as e:
            logger.warning(f"Error loading existing results: {e}")
    return pd.DataFrame()


def save_results_incremental(responses, csv_path, lock=None):
    """결과를 점진적으로 저장 (멀티프로세싱 안전)"""
    try:
        if lock:
            with lock:
                _save_results_safely(responses, csv_path)
        else:
            _save_results_safely(responses, csv_path)
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def _save_results_safely(responses, csv_path):
    """실제 저장 로직"""
    df_new = pd.DataFrame(responses)
    
    # 기존 파일이 있으면 로드하고 합치기
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        # 기존 데이터에서 현재 카테고리 데이터 제거 (덮어쓰기 위해)
        current_category = df_new['category'].iloc[0] if not df_new.empty else None
        if current_category:
            df_existing = df_existing[df_existing['category'] != current_category]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df_new)} new records for category to {csv_path} (Total: {len(df_combined)} records)")


def get_completed_categories(csv_path, min_records=10):
    """완료된 카테고리 목록 반환"""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                category_counts = df['category'].value_counts()
                completed = []
                for category, count in category_counts.items():
                    logger.info(f"Found category {category} with {count} records")
                    if count >= min_records:
                        completed.append(category)
                logger.info(f"Completed categories: {completed}")
                return completed
        except Exception as e:
            logger.warning(f"Error reading completed categories: {e}")
    return []


def process_batch_streaming(batch_data, model_config, template_type="basic"):
    """스트리밍 방식으로 배치 처리"""
    try:
        # 각 프로세스에서 별도의 LLM 클라이언트 생성
        llm, _ = get_llm_client(
            model_config['provider'], 
            model_config.get('hf_model_id', 'microsoft/Phi-3.5-mini-instruct'),
            model_config['temperature'], 
            model_config['max_tokens'], 
            model_config['max_retries'],
            model_config.get('wait_time', 1.0)
        )
        
        prompt_template = get_prompt_template(template_type)
        chain = prompt_template | llm | MultipleChoicesFiveParser()
        
        results = []
        batch_size = model_config['batch_size']
        max_retries = model_config['max_retries']
        
        for i in range(0, len(batch_data), batch_size):
            mini_batch = batch_data[i:i + batch_size]
            retries = 0
            
            while retries <= max_retries:
                try:
                    preds = chain.batch(mini_batch, {"max_concurrency": batch_size})
                    
                    for qna, pred in zip(mini_batch, preds):
                        results.append({
                            "category": qna["category"],
                            "answer": qna["answer"],
                            "pred": pred[0],
                            "response": pred[1],
                        })
                    break
                    
                except RateLimitError as e:
                    delay = (retries + 1) * 30
                    logger.warning(f"Rate limit error, retrying in {delay} seconds...")
                    time.sleep(delay)
                    retries += 1
                    
                    if retries > max_retries:
                        logger.error(f"Max retries reached for batch")
                        for qna in mini_batch:
                            results.append({
                                "category": qna["category"],
                                "answer": qna["answer"],
                                "pred": "FAILED",
                                "response": "RATE_LIMIT_ERROR",
                            })
                        break
                        
                except openai.BadRequestError as e:
                    logger.error(f"BadRequestError: {e}. Adding failed responses for this batch.")
                    logger.info(f"Question sample: {batch_data[i]['question'][:100]}..." if batch_data else "No question data")
                    # 실패한 질문들에 대해 기본값으로 추가
                    for qna in mini_batch:
                        results.append({
                            "category": qna["category"],
                            "answer": qna["answer"],
                            "pred": "FAILED",
                            "response": "BAD_REQUEST_ERROR",
                        })
                    break
                        
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    for qna in mini_batch:
                        results.append({
                            "category": qna["category"],
                            "answer": qna["answer"],
                            "pred": "FAILED",
                            "response": f"ERROR: {str(e)}",
                        })
                    break
        
        return results
        
    except Exception as e:
        logger.error(f"Error in process_batch_streaming: {e}")
        return []


def process_category_streaming(category_info):
    """단일 카테고리를 스트리밍 방식으로 처리"""
    category, model_config, is_debug, num_debug_samples, template_type, csv_path = category_info
    
    logger.info(f"Processing category {category} in process {os.getpid()}")
    
    try:
        # 카테고리별 데이터셋 로드
        ds = load_dataset("HAERAE-HUB/HAE_RAE_BENCH_1.0", category)["test"]
        df = ds.to_pandas()
        df["category"] = category
        category_ds = Dataset.from_pandas(df)
        
        if is_debug:
            category_ds = category_ds.select(range(min(num_debug_samples, len(category_ds))))
        
        # 스트리밍 처리를 위한 청크 크기 설정
        chunk_size = model_config['batch_size'] * 10
        total_items = len(category_ds)
        category_responses = []
        
        logger.info(f"Processing {total_items} items for category {category} in chunks of {chunk_size}")
        
        # 청크별 진행률 표시 추가
        total_chunks = (total_items + chunk_size - 1) // chunk_size
        with tqdm(total=total_chunks, desc=f"Processing {category}", position=0, leave=True) as pbar:
            # 청크별로 스트리밍 처리
            for start_idx in range(0, total_items, chunk_size):
                end_idx = min(start_idx + chunk_size, total_items)
                chunk_ds = category_ds.select(range(start_idx, end_idx))
                
                # 청크를 배치 데이터로 변환
                chunk_batch = [
                    {
                        "category": category,
                        "question": get_prompt(x),
                        "answer": get_answer(x),
                    }
                    for x in chunk_ds
                ]
                
                # 청크 처리
                chunk_results = process_batch_streaming(chunk_batch, model_config, template_type)
                category_responses.extend(chunk_results)
                
                # 진행률 업데이트
                pbar.update(1)
                pbar.set_postfix({
                    'items': f"{len(category_responses)}/{total_items}",
                    'chunk_size': len(chunk_results)
                })
                
                # 중간 저장 (메모리 절약)
                if len(category_responses) >= 1000:
                    save_results_incremental(category_responses, csv_path)
                    logger.info(f"Intermediate save for category {category}: {len(category_responses)} items")
                    category_responses = []
        
        # 마지막 배치 저장
        if category_responses:
            save_results_incremental(category_responses, csv_path)
        
        logger.info(f"Completed category {category}")
        return category, "completed"
        
    except Exception as e:
        logger.error(f"Error processing category {category}: {e}")
        return category, f"error: {str(e)}"


def benchmark_multiprocess(args):
    """멀티프로세싱을 사용한 벤치마크 실행"""
    
    is_debug = args.is_debug
    
    # 모델 설정
    model_name = os.getenv("MODEL_NAME", "gpt-5-mini")
    model_version = os.getenv("MODEL_VERSION", "2025-08-08")
    
    model_config = {
        'provider': args.model_provider,
        'hf_model_id': args.hf_model_id,
        'batch_size': args.batch_size,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'max_retries': args.max_retries,
        'wait_time': args.wait_time,
    }
    
    # CSV 파일 경로 설정
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/[HAERAE] {model_name}-{model_version}.csv"
    
    # 모든 HAERAE 카테고리 목록
    all_haerae_categories = [
        "General Knowledge",
        "History",
        "Loan Words",
        "Rare Words",
        "Reading Comprehension",
        "Standard Nomenclature",
    ]
    
    # 실행할 카테고리 결정
    if args.categories:
        # 사용자가 지정한 카테고리들 검증
        invalid_categories = [c for c in args.categories if c not in all_haerae_categories]
        if invalid_categories:
            logger.error(f"Invalid categories specified: {invalid_categories}")
            logger.error(f"Available categories: {all_haerae_categories}")
            return
        selected_categories = args.categories
        logger.info(f"🎯 Processing user-specified categories: {selected_categories}")
    else:
        selected_categories = all_haerae_categories
        logger.info(f"🎯 Processing all categories: {selected_categories}")
    
    # 완료된 카테고리 확인
    completed_categories = get_completed_categories(csv_path)
    
    # 남은 카테고리 필터링 (완료되지 않은 카테고리만 처리)
    remaining_categories = [c for c in selected_categories if c not in completed_categories]
    
    if not remaining_categories:
        logger.info("All specified categories already completed!")
        return
    
    logger.info(f"Processing {len(remaining_categories)} remaining categories: {remaining_categories}")
    logger.info(f"Using multiprocessing with {args.max_workers} workers")
    
    # 멀티프로세싱 작업 준비
    category_tasks = [
        (category, model_config, is_debug, args.num_debug_samples, args.template_type, csv_path)
        for category in remaining_categories
    ]
    
    start_time = time.time()
    
    # 멀티프로세싱 실행 - tqdm 개선
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_category = {
            executor.submit(process_category_streaming, task): task[0] 
            for task in category_tasks
        }
        
        completed_count = 0
        # 진행률 표시 개선
        with tqdm(total=len(remaining_categories), desc="Categories", position=1, leave=True) as category_pbar:
            for future in future_to_category:
                category = future_to_category[future]
                try:
                    result_category, status = future.result()
                    completed_count += 1
                    logger.info(f"Category {result_category} completed: {status} ({completed_count}/{len(remaining_categories)})")
                    
                    # 카테고리 진행률 업데이트
                    category_pbar.update(1)
                    category_pbar.set_postfix({
                        'current': result_category,
                        'status': status
                    })
                    
                except Exception as e:
                    logger.error(f"Category {category} failed with exception: {e}")
                    completed_count += 1
                    category_pbar.update(1)
                    category_pbar.set_postfix({
                        'current': category,
                        'status': 'failed'
                    })

    end_time = time.time()
    total_time = format_timespan(end_time - start_time)
    
    logger.info(f"====== [DONE] All specified categories processed in {total_time} =====")
    
    # 최종 평가
    logger.info(f"====== [START] Final Evaluation - CSV_PATH: {csv_path} =====")
    evaluate(csv_path)
    logger.info(f"====== [END] Evaluation completed =====")


def benchmark_sequential(args):
    """기존 순차 처리 방식"""
    is_debug = args.is_debug
    max_retries = args.max_retries
    delay_increment = 30

    num_debug_samples = args.num_debug_samples
    batch_size = args.batch_size
    max_tokens = args.max_tokens
    temperature = args.temperature
    llm, model_name = get_llm_client(
        args.model_provider, args.hf_model_id, temperature, max_tokens, max_retries, args.wait_time
    )
    model_version = (
        os.getenv("MODEL_VERSION")
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
    
    # 전체 평균 계산 (kmmlu_main.py와 동일한 방식)
    overall_avg = result["correct"].mean()
    print(f"Overall Average: {overall_avg}")

    os.makedirs("evals", exist_ok=True)
    filename = csv_path.split("/")[-1].split(".")[0]
    category_avg.to_csv(f"evals/{filename}-eval.csv", index=False)


if __name__ == "__main__":
    dotenv_path = os.getenv('DOTENV_PATH', '.env')
    load_dotenv(dotenv_path, override=True)
    parser = argparse.ArgumentParser(description="HAERAE Benchmark with Multiprocessing and Streaming")

    parser.add_argument("--is_debug", type=str2bool, default=True)
    parser.add_argument("--num_debug_samples", type=int, default=20)
    parser.add_argument("--model_provider", type=str, default="azureopenai")
    parser.add_argument("--hf_model_id", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--template_type", type=str, default="chat")
    
    # 특정 카테고리들만 실행하기 위한 인수
    parser.add_argument(
        "--categories", 
        type=str, 
        nargs='*', 
        default=None, 
        help='Specific categories to process (e.g., --categories "History" "Loan Words")'
    )
    
    # 새로운 멀티프로세싱 관련 인수
    parser.add_argument("--use_multiprocessing", type=str2bool, default=True, help="Enable multiprocessing")
    parser.add_argument("--max_workers", type=int, default=3, help="Maximum number of worker processes")
    parser.add_argument("--wait_time", type=float, default=1.0, help="Wait time between Bedrock requests to avoid throttling")

    args = parser.parse_args()
    valid_providers = ["azureopenai", "openai", "azureml", "azureai", "huggingface", "bedrock"]
    assert (
        args.model_provider in valid_providers
    ), f"Invalid 'model_provider' value. Please choose from {valid_providers}."

    valid_template_types = ["basic", "chat", "gpt5"]
    assert (
        args.template_type in valid_template_types
    ), f"Invalid 'template_type' value. Please choose from {valid_template_types}."

    # 카테고리 인수 로깅
    if args.categories:
        logger.info(f"🎯 User specified categories: {args.categories}")
    else:
        logger.info(f"🎯 Will process all available categories")

    logger.info(args)
    
    # 멀티프로세싱 사용 여부에 따라 실행 방식 선택
    if args.use_multiprocessing and args.max_workers > 1:
        benchmark_multiprocess(args)
    else:
        benchmark_sequential(args)
