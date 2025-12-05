"""CLIcK benchmark evaluation"""
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

from config.question_templates import TYPE_1, TYPE_2, TYPE_3, TYPE_4
from core.evaluator import CLIcKEvaluator
from core.logger import logger
from util.custom_parser import MultipleChoicesFiveParser
from util.common_helper import str2bool, format_timespan, get_provider_name, check_existing_csv_in_debug
from util.evaluate_helper import evaluate


def get_prompt(x):
    """프롬프트 생성"""
    num_choices = len(x["choices"])
    choices = x["choices"]
    
    if num_choices == 4:
        template = TYPE_1 if x["paragraph"] else TYPE_2
        return template.format(
            CONTEXT=x["paragraph"], QUESTION=x["question"],
            A=choices[0], B=choices[1], C=choices[2], D=choices[3]
        ) if x["paragraph"] else template.format(
            QUESTION=x["question"], A=choices[0], B=choices[1], C=choices[2], D=choices[3]
        )
    elif num_choices == 5:
        template = TYPE_3 if x["paragraph"] else TYPE_4
        return template.format(
            CONTEXT=x["paragraph"], QUESTION=x["question"],
            A=choices[0], B=choices[1], C=choices[2], D=choices[3], E=choices[4]
        ) if x["paragraph"] else template.format(
            QUESTION=x["question"], A=choices[0], B=choices[1], C=choices[2], D=choices[3], E=choices[4]
        )
    else:
        raise ValueError(f"Invalid number of choices: {num_choices}")


def get_answer(x):
    """정답 추출"""
    answer_idx = [c.strip() for c in x["choices"]].index(x["answer"].strip())
    return chr(0x41 + answer_idx)


def process_category(category_info):
    """카테고리 처리"""
    category, model_config, is_debug, num_debug_samples, template_type, csv_path = category_info
    
    logger.info(f"Processing category: {category}")
    
    try:
        # 데이터 로드
        click_ds = load_dataset("EunsuKim/CLIcK", split="train")
        with open("mapping/id_to_category.json", "r") as f:
            id_to_category = json.load(f)
        
        # 카테고리 필터링
        category_data = [
            {
                "id": item["id"],
                "category": category,
                "question": get_prompt(item),
                "answer": get_answer(item),
            }
            for item in click_ds
            if id_to_category.get(str(item["id"])) == category
        ]
        
        if is_debug:
            category_data = category_data[:num_debug_samples]
        
        # 평가 실행
        evaluator = CLIcKEvaluator(model_config, template_type)
        results = evaluator.process_batch(category_data, MultipleChoicesFiveParser)
        
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
    
    # .env에서 MODEL_PROVIDER 읽기 (없으면 args 사용)
    model_provider = os.getenv("MODEL_PROVIDER", args.model_provider)
    
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
    
    # 카테고리 목록
    all_categories = [
        "Economy", "Geography", "History", "Law", "Politics",
        "Pop Culture", "Society", "Tradition", "Functional", "Grammar", "Textual"
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
        # 기존 결과에서 완료된 카테고리 확인
        with open("mapping/id_to_category.json", "r") as f:
            id_to_category = json.load(f)
        
        # 각 카테고리의 실제 문제 수 계산
        click_ds = load_dataset("EunsuKim/CLIcK", split="train")
        category_sizes = {}
        for item in click_ds:
            cat = id_to_category.get(str(item["id"]))
            if cat:
                category_sizes[cat] = category_sizes.get(cat, 0) + 1
        
        completed_categories = []
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty:
                df['category'] = df['id'].astype(str).map(id_to_category)
                category_counts = df['category'].value_counts()
                # 실제 카테고리 크기와 비교
                completed_categories = [
                    cat for cat, count in category_counts.items() 
                    if cat in category_sizes and count >= category_sizes[cat]
                ]
                logger.info(f"Completed categories: {completed_categories}")
        
        categories_to_run = [c for c in all_categories if c not in completed_categories]
    
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
    evaluate(csv_path, dataset="CLIcK", verbose=True)


if __name__ == "__main__":
    main()
