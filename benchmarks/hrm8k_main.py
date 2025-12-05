"""HRM8K benchmark evaluation"""
import os
import sys
import argparse
from dotenv import load_dotenv
from datasets import load_dataset
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.evaluator import HRM8KEvaluator
from core.logger import logger
from util.common_helper import str2bool, get_provider_name, check_existing_csv_in_debug
from util.evaluate_helper import evaluate
from util.hrm8k_calculator import extract_answer

def get_prompt(x):
    return f"질문: {x['question']}\n\n위 문제를 단계별로 풀이하고, 최종 답을 #### 뒤에 숫자만 작성하세요."

def get_answer(x):
    return x["answer"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_debug", type=str2bool, default=True)
    parser.add_argument("--num_debug_samples", type=int, default=20)
    parser.add_argument("--model_provider", type=str, default="azureopenai")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--template_type", type=str, default="basic")
    parser.add_argument("--wait_time", type=float, default=1.0)
    parser.add_argument("--subset", type=str, default=None)
    args = parser.parse_args()

    load_dotenv()
    
    model_provider = os.getenv("MODEL_PROVIDER", args.model_provider)
    
    model_config = {
        "provider": model_provider,
        "model_name": os.getenv("MODEL_NAME"),
        "model_version": os.getenv("MODEL_VERSION"),
        "hf_model_id": os.getenv("HF_MODEL_ID"),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_retries": args.max_retries,
        "wait_time": args.wait_time,
        "batch_size": args.batch_size
    }

    model_name = os.getenv("MODEL_NAME", "unknown")
    model_version = os.getenv("MODEL_VERSION", "unknown")
    csv_path = f"./results/[HRM8K] {model_name}-{model_version}.csv"

    # 디버그 모드에서 기존 CSV 확인
    if check_existing_csv_in_debug(csv_path, args.is_debug):
        evaluate(csv_path, dataset="hrm8k", verbose=True)
        return

    all_subsets = ["GSM8K", "MATH", "OMNI_MATH", "MMMLU", "KSM"]
    
    if args.subset:
        subsets_to_run = [args.subset] if args.subset in all_subsets else []
        if not subsets_to_run:
            logger.error(f"Invalid subset: {args.subset}. Choose from {all_subsets}")
            return
    else:
        subsets_to_run = all_subsets

    logger.info(f"🚀 Starting HRM8K evaluation")
    logger.info(f"Model: {get_provider_name(model_provider)}")
    logger.info(f"Debug: {args.is_debug}, Batch: {args.batch_size}")
    logger.info(f"Subsets: {subsets_to_run}")

    evaluator = HRM8KEvaluator(model_config, args.template_type)
    
    for subset in subsets_to_run:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing subset: {subset}")
        
        dataset = load_dataset("HAERAE-HUB/HRM8K", subset, split="test")
        
        if args.is_debug:
            dataset = dataset.select(range(min(args.num_debug_samples, len(dataset))))
            logger.info(f"🔍 Debug mode: {len(dataset)} samples")

        batch_data = [{
            "question": get_prompt(x),
            "answer": get_answer(x),
            "subset": subset,
            "index": i
        } for i, x in enumerate(dataset)]

        responses = evaluator.process_batch(batch_data, None)
        
        for r in responses:
            pred_answer = extract_answer(r["response"])
            r["pred"] = pred_answer
            if pred_answer is not None and r["answer"] is not None:
                try:
                    r["correct"] = abs(float(pred_answer) - float(r["answer"])) < 1e-5
                except (ValueError, TypeError):
                    r["correct"] = False
            else:
                r["correct"] = False
        
        evaluator.save_results(responses, csv_path, merge_key='subset')
        
        accuracy = sum(r["correct"] for r in responses) / len(responses) * 100
        logger.info(f"✅ {subset} Accuracy: {accuracy:.2f}%")

    logger.info(f"\n{'='*50}")
    logger.info("Calculating overall accuracy...")
    evaluate(csv_path, "hrm8k", verbose=True)

if __name__ == "__main__":
    main()
