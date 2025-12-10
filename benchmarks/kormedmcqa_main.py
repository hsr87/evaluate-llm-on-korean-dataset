"""KorMedMCQA benchmark evaluation"""
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
    template = get_question_template(num_choices=5, with_context=False)
    
    return template.format(
        QUESTION=x["question"],
        A=x["A"],
        B=x["B"],
        C=x["C"],
        D=x["D"],
        E=x["E"]
    )


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_debug", type=str2bool, default=True)
    parser.add_argument("--num_debug_samples", type=int, default=20)
    parser.add_argument("--model_provider", type=str, default="azureopenai")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--template_type", type=str, default="basic")
    parser.add_argument("--wait_time", type=float, default=float(os.getenv("WAIT_TIME", "30.0")))
    parser.add_argument("--hf_model_id", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    dotenv_path = os.getenv('DOTENV_PATH', '.env')
    print(f"DEBUG: Loading dotenv from: {dotenv_path}")
    load_dotenv(dotenv_path, override=True)

    # Model config
    model_provider = os.getenv("MODEL_PROVIDER", args.model_provider)
    provider_name = get_provider_name(model_provider)
    logger.info(f"Model Provider: {provider_name}")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    model_version = os.getenv("MODEL_VERSION", "2024-07-18")
    print(f"DEBUG: MODEL_NAME={model_name}, MODEL_VERSION={model_version}")
    
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
    csv_path = f"results/[KorMedMCQA] {model_name}-{model_version}.csv"
    
    # 디버그 모드에서 기존 CSV 확인
    if check_existing_csv_in_debug(csv_path, args.is_debug):
        evaluate(csv_path, dataset="KorMedMCQA", verbose=True)
        return
    
    # 전체 데이터 로드
    logger.info("Loading all data...")
    subsets = ["doctor", "nurse", "pharm", "dentist"]
    all_data = []
    
    for subset in tqdm(subsets, desc="Loading subsets"):
        try:
            ds = load_dataset("sean0042/KorMedMCQA", name=subset)["test"]
            for i, item in enumerate(ds):
                answer = chr(64 + item["answer"])
                all_data.append({
                    "id": f"{subset}_{i}",
                    "category": subset,
                    "question": get_prompt(item),
                    "answer": answer,
                })
        except Exception as e:
            logger.warning(f"Failed to load {subset}: {e}")
    
    # 디버그 모드
    if args.is_debug:
        all_data = all_data[:args.num_debug_samples]
    
    if not all_data:
        logger.info("✅ All data completed!")
        evaluate(csv_path, dataset="KorMedMCQA", verbose=True)
        return
    
    logger.info(f"🚀 [{model_name}] Processing {len(all_data)} samples with {args.num_workers} workers")
    
    # 데이터를 worker 수만큼 균등 분할
    chunk_size = (len(all_data) + args.num_workers - 1) // args.num_workers
    chunks = [
        (i, all_data[i*chunk_size:(i+1)*chunk_size], model_config, args.template_type, csv_path, model_name)
        for i in range(args.num_workers)
        if i*chunk_size < len(all_data)
    ]
    
    # 병렬 실행
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)}
        
        with tqdm(total=len(all_data), desc="Processing samples", unit="samples") as pbar:
            for future in as_completed(futures):
                chunk_idx = futures[future]
                chunk_size = len(chunks[chunk_idx][1])
                pbar.update(chunk_size)
                pbar.set_postfix({"chunk": f"{sum(1 for f in futures if f.done())}/{len(chunks)}"})
        
        results = [future.result() for future in futures]
    
    
    elapsed_time = time.time() - start_time
    logger.info(f"⏱️  Total time: {format_timespan(elapsed_time)}")
    
    # 결과 요약
    for chunk_id, status in results:
        if status == "completed":
            logger.info(f"✅ Chunk {chunk_id}: {status}")
        else:
            logger.error(f"❌ Chunk {chunk_id}: {status}")
    
    # 최종 평가
    evaluate(csv_path, dataset="KorMedMCQA", verbose=True)


if __name__ == "__main__":
    main()
