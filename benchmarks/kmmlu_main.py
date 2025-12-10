"""KMMLU benchmark evaluation"""
import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.question_templates import get_question_template
from core.evaluator import KMMLUEvaluator
from core.logger import logger
from util.custom_parser import MultipleChoicesFourParser
from util.common_helper import str2bool, format_timespan, get_provider_name, check_existing_csv_in_debug
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
    template = get_question_template(num_choices=4, with_context=False, few_shot=bool(few_shots))
    
    format_dict = {
        "QUESTION": x["question"],
        "A": x["A"], "B": x["B"], "C": x["C"], "D": x["D"]
    }
    
    if few_shots:
        format_dict["FEW_SHOTS"] = few_shots
    
    return template.format(**format_dict)


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


def process_chunk(chunk_info):
    """데이터 청크 처리"""
    chunk_id, data_chunk, model_config, template_type, csv_path, model_name = chunk_info
    
    logger.info(f"[{model_name}] Processing chunk {chunk_id} with {len(data_chunk)} samples")
    
    try:
        evaluator = KMMLUEvaluator(model_config, template_type)
        results = evaluator.process_batch(data_chunk, MultipleChoicesFourParser, num_choices=4, csv_path=csv_path, chunk_id=chunk_id)
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
    parser.add_argument("--is_hard", type=str2bool, default=False)
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    load_dotenv(os.getenv('DOTENV_PATH', '.env'), override=True)
    
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
    
    # 디버그 모드에서 기존 CSV 확인
    if check_existing_csv_in_debug(csv_path, args.is_debug):
        evaluate(csv_path, dataset=dataset_label, verbose=True)
        return
    
    # 카테고리 목록
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
    
    # 카테고리 필터링
    if args.categories:
        invalid = [c for c in args.categories if c not in all_categories]
        if invalid:
            logger.error(f"Invalid categories: {invalid}")
            return
        all_categories = args.categories
    
    # 전체 데이터 로드
    logger.info("Loading all data...")
    all_data = []
    
    for category in tqdm(all_categories, desc="Loading categories"):
        try:
            dataset_category = map_category_name(category, args.is_hard)
            ds_dict = load_dataset(hf_dataset_id, dataset_category)
            
            # Few-shot 프롬프트
            few_shots_prompt = None
            if few_shots_config['enabled']:
                dev_df = ds_dict["dev"].to_pandas()
                dev_df["answer"] = dev_df["answer"].apply(map_answer)
                few_shots_prompt = generate_few_shots_prompt(dev_df.head(few_shots_config['num_shots']))
            
            # 테스트 데이터
            test_df = ds_dict["test"].to_pandas()
            test_df["answer"] = test_df["answer"].apply(map_answer)
            test_df["category"] = category
            
            for _, row in test_df.iterrows():
                all_data.append({
                    "category": category,
                    "question": get_prompt(row, few_shots_prompt),
                    "answer": get_answer(row),
                })
        except Exception as e:
            logger.warning(f"Failed to load {category}: {e}")
    
    # 디버그 모드
    if args.is_debug:
        all_data = all_data[:args.num_debug_samples]
    
    # 기존 완료된 데이터 제외 (category 기반)
    import pandas as pd
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if not df.empty:
            completed_cats = df.groupby('category').size().to_dict()
            logger.info(f"Found {len(completed_cats)} completed categories")
    
    if not all_data:
        logger.info("✅ All data completed!")
        evaluate(csv_path, dataset=dataset_label, verbose=True)
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
            for future in as_completed(futures):
                chunk_idx = futures[future]
                chunk_size = len(chunks[chunk_idx][1])
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
    evaluate(csv_path, dataset=dataset_label, verbose=True)


if __name__ == "__main__":
    main()
