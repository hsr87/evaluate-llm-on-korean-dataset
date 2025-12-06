"""KorMedMCQA benchmark evaluation"""
import os
import sys
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
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


def process_subset(subset_info):
    """서브셋 처리"""
    subset, model_config, is_debug, num_debug_samples, template_type, csv_path = subset_info
    
    logger.info(f"Processing subset: {subset}")
    
    try:
        # 데이터셋 로드
        ds = load_dataset("sean0042/KorMedMCQA", name=subset)["test"]
        
        # 서브셋 데이터 준비
        subset_data = []
        for i, item in enumerate(ds):
            if is_debug and i >= num_debug_samples:
                break
            
            # Convert answer from 1-5 to A-E
            answer = chr(64 + item["answer"])
            
            subset_data.append({
                "id": f"{subset}_{i}",
                "category": subset,
                "question": get_prompt(item),
                "answer": answer,
            })
        
        # 평가 실행
        evaluator = CLIcKEvaluator(model_config, template_type)
        results = evaluator.process_batch(subset_data, MultipleChoicesFiveParser, num_choices=5)
        
        # 결과 저장
        evaluator.save_results(results, csv_path)
        
        logger.info(f"✅ Completed subset: {subset}")
        return subset, "completed"
        
    except Exception as e:
        logger.error(f"❌ Error processing {subset}: {e}")
        return subset, f"error: {str(e)}"


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
    parser.add_argument("--wait_time", type=float, default=1.0)
    parser.add_argument("--hf_model_id", type=str, default=None)
    parser.add_argument("--max_workers", type=int, default=4)
    args = parser.parse_args()

    # Model config
    model_provider = os.getenv("MODEL_PROVIDER", args.model_provider)
    provider_name = get_provider_name(model_provider)
    logger.info(f"Model Provider: {provider_name}")
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
    csv_path = f"results/[KorMedMCQA] {model_name}-{model_version}.csv"
    
    # 디버그 모드에서 기존 CSV 확인
    if check_existing_csv_in_debug(csv_path, args.is_debug):
        evaluate(csv_path, dataset="KorMedMCQA", verbose=True)
        return
    
    # 서브셋 목록
    subsets = ["doctor", "nurse", "pharm", "dentist"]
    
    # 병렬 처리 준비
    subset_tasks = [
        (subset, model_config, args.is_debug, args.num_debug_samples, args.template_type, csv_path)
        for subset in subsets
    ]
    
    # 병렬 실행
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(tqdm(
            executor.map(process_subset, subset_tasks),
            total=len(subset_tasks),
            desc="Processing subsets"
        ))
    
    elapsed_time = time.time() - start_time
    logger.info(f"⏱️  Total time: {format_timespan(elapsed_time)}")
    
    # 결과 요약
    for subset, status in results:
        if status == "completed":
            logger.info(f"✅ {subset}: {status}")
        else:
            logger.error(f"❌ {subset}: {status}")
    
    # 최종 평가
    evaluate(csv_path, dataset="KorMedMCQA", verbose=True)


if __name__ == "__main__":
    main()
