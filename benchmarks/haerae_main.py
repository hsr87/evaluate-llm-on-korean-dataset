"""HAERAE benchmark evaluation"""
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
from core.evaluator import HAERAEEvaluator
from core.logger import logger
from util.custom_parser import MultipleChoicesFiveParser
from util.common_helper import str2bool, format_timespan, get_provider_name, check_existing_csv_in_debug
from util.evaluate_helper import evaluate


def get_prompt(x):
    """프롬프트 생성"""
    template = get_question_template(num_choices=5, with_context=False)
    return template.format(
        QUESTION=x["question"],
        A=x["a"], B=x["b"], C=x["c"], D=x["d"], E=x["e"]
    )


def get_answer(x):
    """정답 추출"""
    return x["answer"].upper().strip()


def process_chunk(chunk_info):
    """데이터 청크 처리"""
    chunk_id, data_chunk, model_config, template_type, csv_path, model_name = chunk_info
    
    logger.info(f"[{model_name}] Processing chunk {chunk_id} with {len(data_chunk)} samples")
    
    try:
        evaluator = HAERAEEvaluator(model_config, template_type)
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
    
    # CSV 경로
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/[HAERAE] {model_name}-{model_version}.csv"
    
    # 디버그 모드에서 기존 CSV 확인
    if check_existing_csv_in_debug(csv_path, args.is_debug):
        evaluate(csv_path, dataset="HAERAE", verbose=True)
        return
    
    # 카테고리 목록
    all_categories = [
        "General Knowledge", "History", "Loan Words",
        "Rare Words", "Reading Comprehension", "Standard Nomenclature"
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
            ds = load_dataset("HAERAE-HUB/HAE_RAE_BENCH_1.0", category)["test"]
            for item in ds:
                all_data.append({
                    "category": category,
                    "question": get_prompt(item),
                    "answer": get_answer(item),
                })
        except Exception as e:
            logger.warning(f"Failed to load {category}: {e}")
    
    # 디버그 모드
    if args.is_debug:
        all_data = all_data[:args.num_debug_samples]
    
    if not all_data:
        logger.info("✅ All data completed!")
        evaluate(csv_path, dataset="HAERAE", verbose=True)
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
    evaluate(csv_path, dataset="HAERAE", verbose=True)


if __name__ == "__main__":
    main()
