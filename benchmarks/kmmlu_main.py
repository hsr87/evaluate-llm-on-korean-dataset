"""KMMLU benchmark evaluation"""
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

from config.question_templates import TYPE_2, TYPE_MMLU_FEW_SHOT
from core.evaluator import KMMLUEvaluator
from core.logger import logger
from util.custom_parser import MultipleChoicesFourParser
from util.common_helper import str2bool, format_timespan, get_provider_name
from util.evaluate_helper import evaluate


def generate_few_shots_prompt(data):
    """Few-shot 프롬프트 생성"""
    prompt = ""
    for row in data:
        prompt += f"질문 (Question): {row['question']}\n"
        prompt += f"보기 (Options)\nA: {row['A']}, B: {row['B']}, C: {row['C']}, D: {row['D']}\n"
        prompt += f"정답 (Answer): {row['answer']}\n\n"
    return prompt


def get_prompt(x, few_shots=None):
    """프롬프트 생성"""
    if few_shots is None:
        return TYPE_2.format(
            QUESTION=x["question"],
            A=x["A"], B=x["B"], C=x["C"], D=x["D"]
        )
    else:
        return TYPE_MMLU_FEW_SHOT.format(
            FEW_SHOTS=few_shots,
            QUESTION=x["question"],
            A=x["A"], B=x["B"], C=x["C"], D=x["D"]
        )


def get_answer(x):
    """정답 추출"""
    return x["answer"].upper().strip()


def map_answer(answer):
    """숫자를 문자로 변환"""
    return {1: "A", 2: "B", 3: "C", 4: "D"}[answer]


def map_category_name(snake_case_name, is_hard=False):
    """KMMLU는 Title-Case, KMMLU-HARD는 snake_case"""
    if is_hard:
        return snake_case_name
    return "-".join(word.capitalize() if word != "and" else word for word in snake_case_name.split("_"))


def process_category(category_info):
    """카테고리 처리"""
    category, hf_dataset_id, model_config, few_shots_config, is_debug, num_debug_samples, template_type, csv_path, is_hard = category_info
    
    logger.info(f"Processing category: {category}")
    
    try:
        # 데이터 로드
        dataset_category = map_category_name(category, is_hard)
        ds_dict = load_dataset(hf_dataset_id, dataset_category)
        
        # Few-shot 프롬프트 생성
        few_shots_prompt = None
        if few_shots_config['enabled']:
            dev_ds = ds_dict["dev"]
            dev_df = dev_ds.to_pandas()
            dev_df["answer"] = dev_df["answer"].apply(map_answer)
            few_shots_prompt = generate_few_shots_prompt(dev_df.head(few_shots_config['num_shots']))
        
        # 테스트 데이터 준비
        test_ds = ds_dict["test"]
        test_df = test_ds.to_pandas()
        test_df["answer"] = test_df["answer"].apply(map_answer)
        test_df["category"] = category
        category_ds = Dataset.from_pandas(test_df)
        
        if is_debug:
            category_ds = category_ds.select(range(min(num_debug_samples, len(category_ds))))
        
        # 배치 데이터 준비
        batch_data = [
            {
                "category": category,
                "question": get_prompt(x, few_shots_prompt),
                "answer": get_answer(x),
            }
            for x in category_ds
        ]
        
        # 평가 실행
        evaluator = KMMLUEvaluator(model_config, template_type)
        results = evaluator.process_batch(batch_data, MultipleChoicesFourParser)
        
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
    parser.add_argument("--is_hard", type=str2bool, default=False)
    parser.add_argument("--num_shots", type=int, default=0)
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
    
    few_shots_config = {
        'enabled': args.num_shots > 0,
        'num_shots': args.num_shots,
    }
    
    # 데이터셋 ID
    hf_dataset_id = "HAERAE-HUB/KMMLU-HARD" if args.is_hard else "HAERAE-HUB/KMMLU"
    dataset_label = "KMMLU-HARD" if args.is_hard else "KMMLU"
    shot_label = f"-{args.num_shots}shot" if args.num_shots > 0 else ""
    
    # CSV 경로
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/[{dataset_label}] {model_name}-{model_version}{shot_label}.csv"
    
    # 카테고리 목록 (실제 데이터셋에서 사용 가능한 것만)
    all_categories = [
        "maritime_engineering", "materials_engineering", "railway_and_automotive_engineering",
        "biology", "public_safety", "criminal_law", "information_technology", "geomatics",
        "management", "math", "accounting", "chemistry", "nondestructive_testing",
        "computer_science", "ecology", "health", "political_science_and_sociology", "patent",
        "electrical_engineering", "electronics_engineering", "korean_history",
        "gas_technology_and_engineering", "machine_design_and_manufacturing", "chemical_engineering",
        "telecommunications_and_wireless_technology", "food_processing", "social_welfare",
        "real_estate", "marketing", "mechanical_engineering", "fashion", "psychology",
        "taxation", "environmental_science", "refrigerating_machinery", "education",
        "industrial_engineer", "civil_engineering", "energy_management", "law",
        "agricultural_sciences", "interior_architecture_and_design",
        "aviation_engineering_and_maintenance", "construction", "economics"
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
        evaluator = KMMLUEvaluator(model_config, args.template_type)
        completed = evaluator.get_completed_categories(csv_path, min_records=10)
        categories_to_run = [c for c in all_categories if c not in completed]
    
    if not categories_to_run:
        logger.info("✅ All categories completed!")
        return
    
    logger.info(f"Processing {len(categories_to_run)} categories")
    
    # 멀티프로세싱 실행
    start_time = time.time()
    
    category_tasks = [
        (cat, hf_dataset_id, model_config, few_shots_config, args.is_debug, 
         args.num_debug_samples, args.template_type, csv_path, args.is_hard)
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
    dataset_name = "KMMLU-HARD" if args.is_hard else "KMMLU"
    evaluate(csv_path, dataset=dataset_name, verbose=True)


if __name__ == "__main__":
    main()
