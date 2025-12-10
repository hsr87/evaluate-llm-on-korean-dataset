"""CLIcK benchmark evaluation"""
import os
import sys
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.question_templates import get_question_template
from core.evaluator import CLIcKEvaluator
from core.logger import logger
from util.custom_parser import MultipleChoicesFiveParser
from util.common_helper import str2bool, format_timespan, get_provider_name, check_existing_csv_in_debug
from util.evaluate_helper import evaluate


def get_prompt(x):
    """프롬프트 생성"""
    num_choices = len(x["choices"])
    choices = x["choices"]
    has_context = bool(x["paragraph"])
    
    template = get_question_template(num_choices=num_choices, with_context=has_context)
    
    format_dict = {
        "QUESTION": x["question"],
        **{chr(65 + i): choices[i] for i in range(num_choices)}
    }
    
    if has_context:
        format_dict["CONTEXT"] = x["paragraph"]
    
    return template.format(**format_dict)


def get_answer(x):
    """정답 추출"""
    answer_idx = [c.strip() for c in x["choices"]].index(x["answer"].strip())
    return chr(0x41 + answer_idx)


def process_chunk(chunk_info):
    """데이터 청크 처리"""
    chunk_id, data_chunk, model_config, template_type, csv_path, model_name = chunk_info
    
    logger.info(f"[{model_name}] Processing chunk {chunk_id} with {len(data_chunk)} samples")
    
    try:
        evaluator = CLIcKEvaluator(model_config, template_type)
        results = evaluator.process_batch(data_chunk, MultipleChoicesFiveParser, num_choices=5, csv_path=csv_path, chunk_id=chunk_id)
        # save_results는 이제 process_batch 내에서 실시간으로 처리됨
        
        logger.info(f"✅ [{model_name}] Completed chunk {chunk_id}")
        return chunk_id, "completed"
        
    except Exception as e:
        logger.error(f"❌ Error processing chunk {chunk_id}: {e}")
        return chunk_id, f"error: {str(e)}"


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_debug", type=str2bool, default=True)
    parser.add_argument("--num_debug_samples", type=int, default=5)
    parser.add_argument("--model_provider", type=str, default="azureopenai")
    parser.add_argument("--hf_model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--template_type", type=str, default="basic")
    parser.add_argument("--wait_time", type=float, default=float(os.getenv("WAIT_TIME", "30.0")))
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    dotenv_path = os.getenv('DOTENV_PATH', '.env')
    print(f"DEBUG: Loading dotenv from: {dotenv_path}")
    load_dotenv(dotenv_path, override=True)
    
    # .env에서 MODEL_PROVIDER 읽기 (없으면 args 사용)
    model_provider = os.getenv("MODEL_PROVIDER", args.model_provider)
    print(f"DEBUG: MODEL_NAME={os.getenv('MODEL_NAME')}, MODEL_VERSION={os.getenv('MODEL_VERSION')}")
    
    logger.info(f"Using {get_provider_name(model_provider)} as model provider.")
    
    # 모델 설정
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    model_version = os.getenv("MODEL_VERSION", "2024-07-18")
    
    model_config = {
        'provider': model_provider,
        'hf_model_id': args.hf_model_id,
        'batch_size': args.batch_size,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'max_retries': args.max_retries,
        'wait_time': args.wait_time,
    }
    
    # CSV 경로
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/[CLIcK] {model_name}-{model_version}.csv"
    
    # 디버그 모드에서 기존 CSV 확인
    if check_existing_csv_in_debug(csv_path, args.is_debug):
        evaluate(csv_path, dataset="CLIcK", verbose=True)
        return
    
    # 전체 데이터 로드
    click_ds = load_dataset("EunsuKim/CLIcK", split="train")
    with open("mapping/id_to_category.json", "r") as f:
        id_to_category = json.load(f)
    
    # 전체 데이터 준비
    all_data = [
        {
            "id": item["id"],
            "category": id_to_category.get(str(item["id"])),
            "question": get_prompt(item),
            "answer": get_answer(item),
        }
        for item in click_ds
    ]
    
    # 카테고리 필터링 (지정된 경우)
    if args.categories:
        all_categories = list(set(id_to_category.values()))
        invalid = [c for c in args.categories if c not in all_categories]
        if invalid:
            logger.error(f"Invalid categories: {invalid}")
            logger.error(f"Available: {all_categories}")
            return
        all_data = [d for d in all_data if d["category"] in args.categories]
    
    # 디버그 모드
    if args.is_debug:
        all_data = all_data[:args.num_debug_samples]
    
    # 기존 결과 제외
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if not df.empty:
            completed_ids = set(df['id'].tolist())
            all_data = [d for d in all_data if d['id'] not in completed_ids]
            logger.info(f"Skipping {len(completed_ids)} completed samples")
    
    if not all_data:
        logger.info("✅ All data completed!")
        evaluate(csv_path, dataset="CLIcK", verbose=True)
        return
    
    logger.info(f"🚀 [{model_name}] Processing {len(all_data)} samples with {args.num_workers} workers")
    
    # 데이터를 worker 수만큼 균등 분할
    chunk_size = (len(all_data) + args.num_workers - 1) // args.num_workers
    chunks = [
        (i, all_data[i*chunk_size:(i+1)*chunk_size], model_config, args.template_type, csv_path, model_name)
        for i in range(args.num_workers)
        if i*chunk_size < len(all_data)
    ]
    
    # 멀티프로세싱 실행
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)}
        
        with tqdm(total=len(all_data), desc="Processing samples", unit="samples") as pbar:
            completed_samples = 0
            for future in as_completed(futures):
                chunk_idx = futures[future]
                chunk_size = len(chunks[chunk_idx][1])  # chunks[i][1] is the data chunk
                completed_samples += chunk_size
                pbar.update(chunk_size)
                pbar.set_postfix({"chunk": f"{sum(1 for f in futures if f.done())}/{len(chunks)}"})
        
        results = [future.result() for future in futures]
    
    
    # 결과 요약
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*50}")
    logger.info(f"✅ [{model_name}] Evaluation completed in {format_timespan(elapsed)}")
    logger.info(f"Results saved to: {csv_path}")
    
    for chunk_id, status in results:
        logger.info(f"  Chunk {chunk_id}: {status}")
    
    # 정확도 평가
    logger.info(f"\n{'='*50}")
    logger.info("Calculating accuracy...")
    evaluate(csv_path, dataset="CLIcK", verbose=True)


if __name__ == "__main__":
    main()
