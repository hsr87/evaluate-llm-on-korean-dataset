"""HAERAE benchmark evaluation"""
import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.question_templates import TYPE_4
from core.evaluator import HAERAEEvaluator
from core.logger import logger
from util.custom_parser import MultipleChoicesFiveParser
from util.common_helper import str2bool, format_timespan, get_provider_name
from util.evaluate_helper import evaluate


def get_prompt(x):
    """프롬프트 생성"""
    return TYPE_4.format(
        QUESTION=x["question"],
        A=x["a"], B=x["b"], C=x["c"], D=x["d"], E=x["e"]
    )


def get_answer(x):
    """정답 추출"""
    return x["answer"].upper().strip()


def process_category(category_info):
    """카테고리 처리"""
    category, model_config, is_debug, num_debug_samples, template_type, csv_path = category_info
    
    logger.info(f"Processing category: {category}")
    
    try:
        # 데이터 로드
        ds = load_dataset("HAERAE-HUB/HAE_RAE_BENCH_1.0", category)["test"]
        df = ds.to_pandas()
        df["category"] = category
        category_ds = Dataset.from_pandas(df)
        
        if is_debug:
            category_ds = category_ds.select(range(min(num_debug_samples, len(category_ds))))
        
        # 배치 데이터 준비
        batch_data = [
            {
                "category": category,
                "question": get_prompt(x),
                "answer": get_answer(x),
            }
            for x in category_ds
        ]
        
        # 평가 실행
        evaluator = HAERAEEvaluator(model_config, template_type)
        results = evaluator.process_batch(batch_data, MultipleChoicesFiveParser)
        
        # 결과 저장
        evaluator.save_results(results, csv_path)
        
        logger.info(f"✅ Completed category: {category}")
        return category, "completed"
        
    except Exception as e:
        logger.error(f"❌ Error processing {category}: {e}")
        return category, f"error: {str(e)}"


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
    parser.add_argument("--wait_time", type=float, default=1.0)
    parser.add_argument("--categories", nargs="+", default=None)
    args = parser.parse_args()
    
    load_dotenv()
    
    logger.info(f"Using {get_provider_name(args.model_provider)} as model provider.")
    
    # 모델 설정
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    model_version = os.getenv("MODEL_VERSION", "2024-07-18")
    
    model_config = {
        'provider': args.model_provider,
        'hf_model_id': args.hf_model_id,
        'batch_size': args.batch_size,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'max_retries': args.max_retries,
        'wait_time': args.wait_time,
    }
    
    # CSV 경로
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/[HAERAE] {model_name}-{model_version}.csv"
    
    # 카테고리 목록
    all_categories = [
        "General Knowledge", "History", "Loan Words",
        "Rare Words", "Reading Comprehension", "Standard Nomenclature"
    ]
    
    # 실행할 카테고리 결정
    if args.categories:
        invalid = [c for c in args.categories if c not in all_categories]
        if invalid:
            logger.error(f"Invalid categories: {invalid}")
            logger.error(f"Available: {all_categories}")
            return
        categories_to_run = args.categories
    else:
        evaluator = HAERAEEvaluator(model_config, args.template_type)
        completed = evaluator.get_completed_categories(csv_path, min_records=10)
        categories_to_run = [c for c in all_categories if c not in completed]
    
    if not categories_to_run:
        logger.info("✅ All categories completed!")
        return
    
    logger.info(f"Processing {len(categories_to_run)} categories: {categories_to_run}")
    
    # 멀티프로세싱 실행
    start_time = time.time()
    
    category_tasks = [
        (cat, model_config, args.is_debug, args.num_debug_samples, args.template_type, csv_path)
        for cat in categories_to_run
    ]
    
    with ProcessPoolExecutor(max_workers=min(4, len(categories_to_run))) as executor:
        results = list(tqdm(
            executor.map(process_category, category_tasks),
            total=len(category_tasks),
            desc="Processing categories",
            position=0
        ))
    
    # 결과 요약
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation completed in {format_timespan(elapsed)}")
    logger.info(f"Results saved to: {csv_path}")
    
    for category, status in results:
        logger.info(f"  {category}: {status}")
    
    # 정확도 평가
    logger.info(f"\n{'='*50}")
    logger.info("Calculating accuracy...")
    evaluate(csv_path, dataset="HAERAE", verbose=True)


if __name__ == "__main__":
    main()
