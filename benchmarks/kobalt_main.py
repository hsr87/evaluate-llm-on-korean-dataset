"""KoBALT benchmark evaluation"""
import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.evaluator import KoBALTEvaluator
from core.logger import logger
from util.evaluate_helper import evaluate
from util.common_helper import check_existing_csv_in_debug, get_provider_name, str2bool, format_timespan
from util.custom_parser import MultipleChoicesTenParser

def process_level(level_info):
    """레벨별 처리"""
    level, model_config, is_debug, num_debug_samples, template_type, csv_path = level_info
    
    level_names = {1: "Easy", 2: "Moderate", 3: "Hard"}
    level_name = level_names[level]
    
    logger.info(f"Processing level: {level_name} (Level {level})")
    
    try:
        dataset = load_dataset("snunlp/KoBALT-700", "kobalt_v1", split="raw")
        level_dataset = dataset.filter(lambda x: x["Level"] == level)
        
        if is_debug:
            level_dataset = level_dataset.select(range(min(num_debug_samples, len(level_dataset))))
        
        logger.info(f"Level {level_name}: {len(level_dataset)} samples")
        
        batch_data = [
            {
                "category": item["Class"],
                "subcategory": item.get("Subclass"),
                "level": item.get("Level"),
                "question": item["Question"],
                "answer": item["Answer"],
            }
            for item in level_dataset
        ]
        
        evaluator = KoBALTEvaluator(model_config, template_type)
        results = evaluator.process_batch(batch_data, MultipleChoicesTenParser, num_choices=10)
        evaluator.save_results(results, csv_path, merge_key='level')
        
        logger.info(f"✅ Completed level: {level_name}")
        return level_name, "completed"
        
    except Exception as e:
        logger.error(f"❌ Error processing {level_name}: {e}")
        return level_name, f"error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="KoBALT Benchmark Evaluation")
    parser.add_argument("--is_debug", type=str2bool, default=True)
    parser.add_argument("--num_debug_samples", type=int, default=5)
    parser.add_argument("--model_provider", type=str, default="azureopenai")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--template_type", type=str, default="basic")
    parser.add_argument("--wait_time", type=float, default=1.0)
    parser.add_argument("--levels", nargs="+", type=int, default=None, help="Filter by levels (1, 2, 3)")
    args = parser.parse_args()
    
    load_dotenv()
    
    model_provider = os.getenv("MODEL_PROVIDER", args.model_provider)
    logger.info(f"Using {get_provider_name(model_provider)} as model provider.")
    
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    model_version = os.getenv("MODEL_VERSION", "2024-07-18")
    
    model_config = {
        'provider': model_provider,
        'batch_size': args.batch_size,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'max_retries': args.max_retries,
        'wait_time': args.wait_time,
    }
    
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/[KoBALT] {model_name}-{model_version}.csv"
    
    if check_existing_csv_in_debug(csv_path, args.is_debug):
        evaluate(csv_path, dataset="KoBALT", verbose=True)
        return
    
    all_levels = [1, 2, 3]
    
    if args.levels:
        invalid = [l for l in args.levels if l not in all_levels]
        if invalid:
            logger.error(f"Invalid levels: {invalid}")
            logger.error(f"Available: {all_levels}")
            return
        levels_to_run = args.levels
    else:
        dataset = load_dataset("snunlp/KoBALT-700", "kobalt_v1", split="raw")
        level_sizes = {
            1: len(dataset.filter(lambda x: x["Level"] == 1)),
            2: len(dataset.filter(lambda x: x["Level"] == 2)),
            3: len(dataset.filter(lambda x: x["Level"] == 3)),
        }
        
        evaluator = KoBALTEvaluator(model_config, args.template_type)
        completed = evaluator.get_completed_categories(csv_path, category_sizes=level_sizes, category_key='level')
        levels_to_run = [l for l in all_levels if l not in completed]
    
    if not levels_to_run:
        logger.info("✅ All levels completed!")
        evaluate(csv_path, dataset="KoBALT", verbose=True)
        return
    
    logger.info(f"Processing {len(levels_to_run)} levels: {levels_to_run}")
    
    start_time = time.time()
    
    level_tasks = [
        (level, model_config, args.is_debug, args.num_debug_samples, args.template_type, csv_path)
        for level in levels_to_run
    ]
    
    with ProcessPoolExecutor(max_workers=min(3, len(levels_to_run))) as executor:
        results = list(tqdm(
            executor.map(process_level, level_tasks),
            total=len(level_tasks),
            desc="Processing levels",
            position=0
        ))
    
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation completed in {format_timespan(elapsed)}")
    logger.info(f"Results saved to: {csv_path}")
    
    for level_name, status in results:
        logger.info(f"  {level_name}: {status}")
    
    logger.info(f"\n{'='*50}")
    logger.info("Calculating accuracy...")
    evaluate(csv_path, dataset="KoBALT", verbose=True)

if __name__ == "__main__":
    main()
