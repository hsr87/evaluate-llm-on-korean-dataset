import os
import json
import time
import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager, Queue
import multiprocessing as mp

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


def _save_results_safely(responses, csv_path):
    """실제 저장 로직"""
    df_new = pd.DataFrame(responses)
    
    # 파일 경로를 절대 경로로 변환
    abs_csv_path = os.path.abspath(csv_path)
    
    # 기존 파일이 있으면 로드하고 합치기
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        # 기존 데이터에서 현재 카테고리 데이터 제거 (덮어쓰기 위해)
        current_category = df_new['category'].iloc[0] if not df_new.empty else None
        if current_category:
            df_existing = df_existing[df_existing['category'] != current_category]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        
        logger.info(f"📁 Updated existing file: {abs_csv_path}")
        logger.info(f"   - Added {len(df_new)} new records for category: {current_category}")
        logger.info(f"   - Total records after merge: {len(df_combined)}")
    else:
        df_combined = df_new
        current_category = df_new['category'].iloc[0] if not df_new.empty else "Unknown"
        logger.info(f"📁 Created new file: {abs_csv_path}")
        logger.info(f"   - Initial records for category {current_category}: {len(df_combined)}")
    
    # 디렉토리 생성 확인
    os.makedirs(os.path.dirname(abs_csv_path), exist_ok=True)
    
    # 파일 저장
    df_combined.to_csv(csv_path, index=False)
    
    # 파일 크기 확인
    file_size = os.path.getsize(csv_path)
    file_size_mb = file_size / (1024 * 1024)
    
    logger.info(f"✅ Successfully saved CSV file:")
    logger.info(f"   - File path: {abs_csv_path}")
    logger.info(f"   - File size: {file_size:,} bytes ({file_size_mb:.2f} MB)")
    logger.info(f"   - Total records: {len(df_combined)}")
    
    # 카테고리별 분포 로깅 (KMMLU의 경우)
    try:
        category_counts = df_combined['category'].value_counts()
        logger.info(f"   - Categories saved: {list(category_counts.index)}")
        logger.info(f"   - Records per category: {dict(category_counts)}")
    except Exception as e:
        logger.warning(f"Could not analyze category distribution: {e}")


def save_results_incremental(responses, csv_path, lock=None):
    """결과를 점진적으로 저장 (멀티프로세싱 안전)"""
    try:
        if lock:
            with lock:
                _save_results_safely(responses, csv_path)
        else:
            _save_results_safely(responses, csv_path)
    except Exception as e:
        logger.error(f"❌ Error saving results to {os.path.abspath(csv_path)}: {e}")
        raise


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


def process_batch_streaming(batch_data, model_config, few_shots_prompt, template_type="basic"):
    """스트리밍 방식으로 배치 처리"""
    try:
        # 각 프로세스에서 별도의 LLM 클라이언트 생성
        llm, _ = get_llm_client(
            model_config['provider'], 
            model_config.get('hf_model_id', 'microsoft/Phi-3.5-mini-instruct'),
            model_config['temperature'], 
            model_config['max_tokens'], 
            model_config['max_retries']
        )
        
        prompt_template = get_prompt_template(template_type)
        chain = prompt_template | llm | MultipleChoicesFourParser()
        
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
    category, hf_dataset_id, model_config, few_shots_config, is_hard, is_debug, num_debug_samples, template_type, csv_path = category_info
    
    logger.info(f"Processing category {category} in process {os.getpid()}")
    
    try:
        # 데이터셋 로드
        ds_dict = load_dataset(hf_dataset_id, category)
        
        # Few-shot 프롬프트 생성
        few_shots_prompt = None
        if few_shots_config['use_few_shots']:
            ds_dev = ds_dict["dev"]
            ds_dev = ds_dev.map(lambda x: {"answer": map_answer(x["answer"])})
            if is_hard:
                ds_dev = ds_dev.map(lambda x: {"category": convert_to_pascal_case(x["category"])})
            else:
                ds_dev = ds_dev.rename_column("Category", "category")
            few_shots_prompt = generate_few_shots_prompt(ds_dev)
        
        # 테스트 데이터 준비
        ds = ds_dict["test"]
        ds = ds.map(lambda x: {"answer": map_answer(x["answer"])})
        if is_hard:
            ds = ds.map(lambda x: {"category": convert_to_pascal_case(x["category"])})
        else:
            ds = ds.rename_column("Category", "category")
        
        if is_debug:
            ds = ds.select(range(num_debug_samples))
        
        # 스트리밍 처리를 위한 청크 크기 설정
        chunk_size = model_config['batch_size'] * 10  # 배치 크기의 10배씩 처리
        total_items = len(ds)
        category_responses = []
        
        logger.info(f"Processing {total_items} items for category {category} in chunks of {chunk_size}")
        
        # 청크별로 스트리밍 처리
        for start_idx in range(0, total_items, chunk_size):
            end_idx = min(start_idx + chunk_size, total_items)
            chunk_ds = ds.select(range(start_idx, end_idx))
            
            # 청크를 배치 데이터로 변환
            chunk_batch = [
                {
                    "category": category,
                    "question": get_prompt(x, few_shots_prompt),
                    "answer": get_answer(x),
                }
                for x in chunk_ds
            ]
            
            # 청크 처리
            chunk_results = process_batch_streaming(chunk_batch, model_config, few_shots_prompt, template_type)
            category_responses.extend(chunk_results)
            
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
    few_shots = "5shot" if args.use_few_shots else "0shot"
    
    if args.is_hard:
        hf_dataset_id = "HAERAE-HUB/KMMLU-HARD"
        dataset_name = "KMMLU-HARD"
        kmmlu_category = [
            "accounting", "agricultural_sciences", "aviation_engineering_and_maintenance",
            "biology", "chemical_engineering", "chemistry", "civil_engineering",
            "computer_science", "construction", "criminal_law", "ecology", "economics",
            "education", "electrical_engineering", "electronics_engineering",
            "energy_management", "environmental_science", "fashion", "food_processing",
            "gas_technology_and_engineering", "geomatics", "health", "industrial_engineer",
            "information_technology", "interior_architecture_and_design", "korean_history",
            "law", "machine_design_and_manufacturing", "management", "maritime_engineering",
            "marketing", "materials_engineering", "math", "mechanical_engineering",
            "nondestructive_testing", "patent", "political_science_and_sociology",
            "psychology", "public_safety", "railway_and_automotive_engineering",
            "real_estate", "refrigerating_machinery", "social_welfare", "taxation",
            "telecommunications_and_wireless_technology",
        ]
    else:
        hf_dataset_id = "HAERAE-HUB/KMMLU"
        dataset_name = "KMMLU"
        kmmlu_category = [
            "Accounting", "Agricultural-Sciences", "Aviation-Engineering-and-Maintenance",
            "Biology", "Chemical-Engineering", "Chemistry", "Civil-Engineering",
            "Computer-Science", "Construction", "Criminal-Law", "Ecology", "Economics",
            "Education", "Electrical-Engineering", "Electronics-Engineering",
            "Energy-Management", "Environmental-Science", "Fashion", "Food-Processing",
            "Gas-Technology-and-Engineering", "Geomatics", "Health", "Industrial-Engineer",
            "Information-Technology", "Interior-Architecture-and-Design", "Korean-History",
            "Law", "Machine-Design-and-Manufacturing", "Management", "Maritime-Engineering",
            "Marketing", "Materials-Engineering", "Math", "Mechanical-Engineering",
            "Nondestructive-Testing", "Patent", "Political-Science-and-Sociology",
            "Psychology", "Public-Safety", "Railway-and-Automotive-Engineering",
            "Real-Estate", "Refrigerating-Machinery", "Social-Welfare", "Taxation",
            "Telecommunications-and-Wireless-Technology",
        ]
    
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
    }
    
    few_shots_config = {
        'use_few_shots': args.use_few_shots
    }
    
    # CSV 파일 경로 설정
    os.makedirs("results", exist_ok=True)
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-5-mini")
    model_version = os.getenv("OPENAI_MODEL_VERSION", "2025-08-08")
    
    # 파일명 생성
    if args.use_few_shots:
        if args.is_hard:
            csv_path = f"results/[KMMLU-HARD] {model_name}-{model_version}-5shot.csv"
        else:
            csv_path = f"results/[KMMLU] {model_name}-{model_version}-5shot.csv"
    else:
        if args.is_hard:
            csv_path = f"results/[KMMLU-HARD] {model_name}-{model_version}-0shot.csv"
        else:
            csv_path = f"results/[KMMLU] {model_name}-{model_version}-0shot.csv"
    
    abs_csv_path = os.path.abspath(csv_path)
    
    logger.info(f"🎯 Target output file: {abs_csv_path}")
    
    # 기존 파일 상태 확인
    if os.path.exists(csv_path):
        file_size = os.path.getsize(csv_path)
        existing_df = pd.read_csv(csv_path)
        logger.info(f"📋 Found existing file with {len(existing_df)} records ({file_size:,} bytes)")
    else:
        logger.info(f"📋 No existing file found - will create new file")
    
    # 완료된 카테고리 확인
    completed_categories = get_completed_categories(csv_path)
    
    # 시작할 카테고리 필터링
    if args.start_category:
        try:
            start_idx = kmmlu_category.index(args.start_category)
            kmmlu_category = kmmlu_category[start_idx:]
            logger.info(f"Starting from category: {args.start_category}")
        except ValueError:
            logger.error(f"Category {args.start_category} not found in category list")
            return
    
    # 완료되지 않은 카테고리만 처리
    remaining_categories = [c for c in kmmlu_category if c not in completed_categories]
    
    if not remaining_categories:
        logger.info("All categories already completed!")
        return
    
    logger.info(f"Processing {len(remaining_categories)} remaining categories: {remaining_categories}")
    logger.info(f"Using multiprocessing with {args.max_workers} workers")
    
    # 멀티프로세싱 작업 준비
    category_tasks = [
        (category, hf_dataset_id, model_config, few_shots_config, args.is_hard, 
         is_debug, args.num_debug_samples, args.template_type, csv_path)
        for category in remaining_categories
    ]
    
    start_time = time.time()
    
    # 멀티프로세싱 실행
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # 진행 상황 모니터링을 위해 submit 사용
        future_to_category = {
            executor.submit(process_category_streaming, task): task[0] 
            for task in category_tasks
        }
        
        completed_count = 0
        for future in tqdm(future_to_category, desc="Processing categories"):
            category = future_to_category[future]
            try:
                result_category, status = future.result()
                completed_count += 1
                logger.info(f"Category {result_category} completed: {status} ({completed_count}/{len(remaining_categories)})")
            except Exception as e:
                logger.error(f"Category {category} failed with exception: {e}")
    
    end_time = time.time()
    total_time = format_timespan(end_time - start_time)
    
    logger.info(f"====== [DONE] All categories processed in {total_time} =====")
    
    # 최종 파일 상태 확인
    if os.path.exists(csv_path):
        final_df = pd.read_csv(csv_path)
        final_file_size = os.path.getsize(csv_path)
        logger.info(f"🏁 Final output file status:")
        logger.info(f"   - Path: {abs_csv_path}")
        logger.info(f"   - Records: {len(final_df)}")
        logger.info(f"   - Size: {final_file_size:,} bytes ({final_file_size/(1024*1024):.2f} MB)")
        
        # 카테고리별 통계
        category_counts = final_df['category'].value_counts()
        logger.info(f"   - Completed categories: {len(category_counts)}")
        logger.info(f"   - Records per category: {dict(category_counts)}")
    else:
        logger.error(f"❌ Final output file not found: {abs_csv_path}")
    
    # 최종 평가
    logger.info(f"====== [START] Final Evaluation - CSV_PATH: {csv_path} =====")
    evaluate(csv_path)
    logger.info(f"====== [END] Evaluation completed =====")


def benchmark_sequential(args):
    """기존 순차 처리 방식"""
    
    logger.info("Using sequential processing")
    # 기존 benchmark 함수 로직을 여기에 복사
    # ... (기존 코드와 동일)


def evaluate_each_category(responses, category):
    df = pd.DataFrame(responses)
    df = df[df["category"] == category]
    # FAILED 응답 필터링
    df = df[df["pred"] != "FAILED"]
    if df.empty:
        return 0.0
    df["correct"] = df["answer"] == df["pred"]
    acc = round(df["correct"].mean() * 100, 2)
    return acc


def evaluate(csv_path):
    abs_csv_path = os.path.abspath(csv_path)
    
    if not os.path.exists(csv_path):
        logger.error(f"❌ CSV file does not exist: {abs_csv_path}")
        return
    
    logger.info(f"📊 Starting evaluation of: {abs_csv_path}")
    
    result = pd.read_csv(csv_path)
    if result.empty:
        logger.error(f"❌ CSV file is empty: {abs_csv_path}")
        return
    
    logger.info(f"📊 Loaded {len(result)} records for evaluation")
    
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
    
    eval_file = f"evals/{filename}-eval.csv"
    category_avg.to_csv(eval_file, index=False)
    
    abs_eval_file = os.path.abspath(eval_file)
    
    logger.info(f"✅ Evaluation results saved:")
    logger.info(f"   - Results file: {abs_eval_file}")
    logger.info(f"   - Categories evaluated: {len(category_avg)}")
    logger.info(f"   - Overall accuracy: {overall_avg:.2%}")
    


if __name__ == "__main__":
    dotenv_path = os.getenv('DOTENV_PATH', '.env')
    load_dotenv(dotenv_path, override=True)
   
    parser = argparse.ArgumentParser(description="KMMLU Benchmark with Multiprocessing and Streaming")

    parser.add_argument("--is_debug", type=str2bool, default=False)
    parser.add_argument("--num_debug_samples", type=int, default=10)
    parser.add_argument("--model_provider", type=str, default="azureopenai")
    parser.add_argument("--hf_model_id", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--template_type", type=str, default="basic")
    parser.add_argument("--is_hard", type=str2bool, default=True)
    parser.add_argument("--use_few_shots", type=str2bool, default=True)
    parser.add_argument("--start_category", type=str, default=None, help="Category to start from (for resuming)")
    
    # 새로운 멀티프로세싱 관련 인수
    parser.add_argument("--use_multiprocessing", type=str2bool, default=True, help="Enable multiprocessing")
    parser.add_argument("--max_workers", type=int, default=3, help="Maximum number of worker processes")
    parser.add_argument("--streaming_chunk_size", type=int, default=1000, help="Chunk size for streaming processing")

    args = parser.parse_args()

    valid_providers = ["azureopenai", "openai", "azureml", "azureai", "huggingface"]
    assert args.model_provider in valid_providers, f"Invalid 'model_provider' value. Please choose from {valid_providers}."

    valid_template_types = ["basic", "chat"]
    assert args.template_type in valid_template_types, f"Invalid 'template_type' value. Please choose from {valid_template_types}."

    logger.info(args)
    
    # 멀티프로세싱 사용 여부에 따라 실행 방식 선택
    if args.use_multiprocessing and args.max_workers > 1:
        benchmark_multiprocess(args)
    else:
        benchmark_sequential(args)