"""Unified evaluator for Korean LLM benchmarks"""
import os
import sys
import time
import json
import pandas as pd
import openai
from openai import RateLimitError
from tqdm import tqdm
from datasets import load_dataset, Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

from .logger import logger
from util.common_helper import get_prompt_template, get_llm_client

class BenchmarkEvaluator:
    """통합 벤치마크 평가 클래스"""
    
    def __init__(self, model_config, template_type="basic"):
        self.model_config = model_config
        self.template_type = template_type
        
    def load_results(self, csv_path):
        """기존 결과 로드"""
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    logger.info(f"Loaded {len(df)} existing records from {csv_path}")
                    return df
            except Exception as e:
                logger.warning(f"Error loading results: {e}")
        return pd.DataFrame()
    
    def save_results(self, responses, csv_path, merge_key='category'):
        """결과 저장 (중복 제거)"""
        df_new = pd.DataFrame(responses)
        
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            current_key = df_new[merge_key].iloc[0] if not df_new.empty else None
            if current_key:
                df_existing = df_existing[df_existing[merge_key] != current_key]
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_combined.to_csv(csv_path, index=False)
        logger.info(f"✅ Saved {len(df_new)} records to {csv_path} (Total: {len(df_combined)})")
        return df_combined
    
    def get_completed_categories(self, csv_path, category_key='category', category_sizes=None):
        """완료된 카테고리 반환 (실제 카테고리 크기와 비교)"""
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if not df.empty and category_key in df.columns:
                    counts = df[category_key].value_counts()
                    if category_sizes:
                        return [cat for cat, cnt in counts.items() if cat in category_sizes and cnt >= category_sizes[cat]]
                    return list(counts.index)
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                logger.warning(f"CSV read error (file may be empty or being written): {e}")
            except Exception as e:
                logger.warning(f"Error reading categories: {e}")
        return []
    
    def process_batch(self, batch_data, parser_class, num_choices=5, use_math_prompt=False):
        """배치 처리
        
        Args:
            batch_data: 처리할 데이터 배치
            parser_class: 파서 클래스
            num_choices: 선택지 개수 (기본값: 5)
            use_math_prompt: 수학 문제용 프롬프트 사용 여부 (기본값: False)
        """
        from tqdm import tqdm
        
        prompt_template, system_prompt = get_prompt_template(
            self.template_type, 
            self.model_config['provider'],
            num_choices=num_choices,
            use_math_prompt=use_math_prompt
        )
        
        # Debug: 첫 번째 샘플의 프롬프트 로깅
        if batch_data and os.getenv("IS_DEBUG", "false").lower() == "true":
            with open("debug_qna.log", "a", encoding="utf-8") as f:
                f.write("\n" + "="*80 + "\n")
                f.write("SYSTEM PROMPT:\n")
                f.write("-"*80 + "\n")
                f.write(system_prompt + "\n")
                f.write("-"*80 + "\n")
                f.write("QUESTION PROMPT (Sample):\n")
                f.write("-"*80 + "\n")
                f.write(batch_data[0]["question"] + "\n")
                f.write("="*80 + "\n\n")
        
        llm, _ = get_llm_client(
            self.model_config['provider'],
            self.model_config.get('hf_model_id'),
            self.model_config['temperature'],
            self.model_config['max_tokens'],
            self.model_config['max_retries'],
            self.model_config.get('wait_time', 1.0),
            system_prompt=system_prompt
        )
        
        chain = prompt_template | llm | parser_class() if parser_class else prompt_template | llm
        
        results = []
        batch_size = self.model_config['batch_size']
        max_retries = self.model_config['max_retries']
        
        with tqdm(total=len(batch_data), desc="  Samples", leave=False, position=1, file=sys.stdout, mininterval=0.1, dynamic_ncols=True) as pbar:
            for i in range(0, len(batch_data), batch_size):
                mini_batch = batch_data[i:i + batch_size]
                
                for retry in range(max_retries + 1):
                    try:
                        # Debug: Parser 없이 raw LLM 응답 먼저 확인
                        if os.getenv("IS_DEBUG", "false").lower() == "true" and parser_class:
                            raw_chain = prompt_template | llm
                            raw_preds = raw_chain.batch(mini_batch, {"max_concurrency": batch_size})
                            with open("debug_qna.log", "a", encoding="utf-8") as f:
                                for qna, raw_pred in zip(mini_batch, raw_preds):
                                    f.write("\n" + "="*80 + "\n")
                                    f.write(f"RAW LLM OUTPUT (before parser):\n")
                                    f.write(f"Type: {type(raw_pred)}\n")
                                    f.write(f"Content: {raw_pred}\n")
                                    if hasattr(raw_pred, 'content'):
                                        f.write(f"Content attr: {repr(raw_pred.content)}\n")
                                    f.write("="*80 + "\n\n")
                        
                        preds = chain.batch(mini_batch, {"max_concurrency": batch_size})
                        
                        # Debug: LLM 응답 로깅
                        if os.getenv("IS_DEBUG", "false").lower() == "true":
                            with open("debug_qna.log", "a", encoding="utf-8") as f:
                                for qna, pred in zip(mini_batch, preds):
                                    f.write("\n" + "="*80 + "\n")
                                    f.write(f"QUESTION: {qna.get('question', 'N/A')}\n")
                                    f.write("-"*80 + "\n")
                                    f.write(f"LLM RESPONSE: {pred}\n")
                                    f.write("="*80 + "\n\n")
                        
                        results.extend([self._make_result(qna, pred) for qna, pred in zip(mini_batch, preds)])
                        pbar.update(len(mini_batch))
                        break
                    except RateLimitError:
                        if retry < max_retries:
                            delay = (retry + 1) * 30
                            logger.warning(f"Rate limit, retrying in {delay}s...")
                            time.sleep(delay)
                        else:
                            results.extend([self._make_failed(qna, "RATE_LIMIT") for qna in mini_batch])
                            pbar.update(len(mini_batch))
                            
                    except openai.BadRequestError as e:
                        error_msg = str(e)
                        
                        # Content filtering 에러 처리
                        if "prompt_filter_results" in error_msg or "Response missing `choices` key" in error_msg:
                            logger.warning(f"Content filtering detected for batch - marking as FILTERED")
                            results.extend([self._make_filtered(qna) for qna in mini_batch])
                            pbar.update(len(mini_batch))
                            break  # 재시도하지 않고 바로 다음 배치로
                        else:
                            logger.error(f"BadRequest: {e}")
                            results.extend([self._make_failed(qna, "BAD_REQUEST") for qna in mini_batch])
                            pbar.update(len(mini_batch))
                            break
                        
                    except KeyError as e:
                        error_msg = str(e)
                        
                        # Content filtering KeyError 처리
                        if "'choices'" in error_msg or "prompt_filter_results" in error_msg:
                            logger.warning(f"Content filtering KeyError - marking batch as FILTERED")
                            for qna in mini_batch:
                                try:
                                    pred = chain.invoke(qna["question"])
                                    results.append(self._make_result(qna, pred))
                                except Exception as individual_error:
                                    logger.error(f"Individual processing failed: {individual_error}")
                                    results.append(self._make_failed(qna, str(individual_error)))
                            pbar.update(len(mini_batch))
                            break  # 재시도하지 않고 바로 다음 배치로
                        else:
                            # 다른 KeyError는 일반 처리
                            logger.error(f"KeyError in batch processing: {e}")
                            if retry < max_retries:
                                time.sleep(2 ** retry)
                            else:
                                results.extend([self._make_failed(qna, f"KEYERROR: {e}") for qna in mini_batch])
                                pbar.update(len(mini_batch))
                            
                    except Exception as e:
                        error_msg = str(e)
                        
                        # Content filtering 일반 Exception 처리
                        if "prompt_filter_results" in error_msg or "Response missing `choices` key" in error_msg:
                            logger.warning(f"Content filtering Exception - marking batch as FILTERED")
                            results.extend([self._make_filtered(qna) for qna in mini_batch])
                            pbar.update(len(mini_batch))
                            break  # 재시도하지 않고 바로 다음 배치로
                        else:
                            logger.error(f"Error processing batch: {e}")
                            if retry < max_retries:
                                logger.warning(f"Retrying...")
                                time.sleep(2 ** retry)
                            else:
                                results.extend([self._make_failed(qna, str(e)) for qna in mini_batch])
                                pbar.update(len(mini_batch))
        
        return results
    
    def _make_result(self, qna, pred):
        """결과 엔트리 생성 (서브클래스에서 오버라이드)"""
        if isinstance(pred, (list, tuple)):
            return {
                "answer": qna["answer"],
                "pred": pred[0],
                "response": pred[1],
            }
        else:
            # Handle AIMessage object
            response_text = pred.content if hasattr(pred, 'content') else str(pred)
            
            # Handle list content (for reasoning models)
            if isinstance(response_text, list):
                response_text = ' '.join([str(item) for item in response_text])
            
            return {
                "answer": qna["answer"],
                "pred": None,
                "response": response_text,
            }
    
    def _make_failed(self, qna, error):
        """실패 엔트리 생성"""
        return {
            "answer": qna["answer"],
            "pred": "FAILED",
            "response": error,
        }
    
    def _make_filtered(self, qna):
        """Content filtering 엔트리 생성"""
        return {
            "answer": qna["answer"],
            "pred": "FILTERED",
            "response": "CONTENT_FILTER_ERROR",
        }


class CLIcKEvaluator(BenchmarkEvaluator):
    """CLIcK 벤치마크 평가"""
    
    def __init__(self, model_config, template_type="basic"):
        super().__init__(model_config, template_type)
        with open("mapping/id_to_category.json", "r") as f:
            self.id_to_category = json.load(f)
    
    def _make_result(self, qna, pred):
        return {
            "id": qna["id"],
            "category": qna.get("category", self.id_to_category.get(str(qna["id"]))),
            "trial": 0,
            "answer": qna["answer"],
            "pred": pred[0],
            "response": pred[1],
        }
    
    def _make_failed(self, qna, error):
        return {
            "id": qna["id"],
            "category": qna.get("category", self.id_to_category.get(str(qna["id"]))),
            "trial": 0,
            "answer": qna["answer"],
            "pred": "FAILED",
            "response": error,
        }
    
    def save_results(self, responses, csv_path):
        """CLIcK는 ID 기준으로 저장"""
        return super().save_results(responses, csv_path, merge_key='id')


class HAERAEEvaluator(BenchmarkEvaluator):
    """HAERAE 벤치마크 평가"""
    
    def _make_result(self, qna, pred):
        return {
            "category": qna["category"],
            "answer": qna["answer"],
            "pred": pred[0],
            "response": pred[1],
        }
    
    def _make_failed(self, qna, error):
        return {
            "category": qna["category"],
            "answer": qna["answer"],
            "pred": "FAILED",
            "response": error,
        }


class KMMLUEvaluator(BenchmarkEvaluator):
    """KMMLU 벤치마크 평가"""
    
    def _make_result(self, qna, pred):
        return {
            "category": qna["category"],
            "answer": qna["answer"],
            "pred": pred[0],
            "response": pred[1],
        }
    
    def _make_failed(self, qna, error):
        return {
            "category": qna["category"],
            "answer": qna["answer"],
            "pred": "FAILED",
            "response": error,
        }


class HRM8KEvaluator(BenchmarkEvaluator):
    """HRM8K 벤치마크 평가"""
    
    def _make_result(self, qna, pred):
        response_text = pred.content if hasattr(pred, 'content') else str(pred)
        if isinstance(response_text, list):
            response_text = ' '.join([str(item) for item in response_text])
        
        return {
            "subset": qna["subset"],
            "index": qna.get("index"),
            "answer": qna["answer"],
            "pred": None,
            "response": response_text,
        }
    
    def _make_failed(self, qna, error):
        return {
            "subset": qna["subset"],
            "index": qna.get("index"),
            "answer": qna["answer"],
            "pred": "FAILED",
            "response": error,
        }


class KoBALTEvaluator(BenchmarkEvaluator):
    """KoBALT 벤치마크 평가"""
    
    def _make_result(self, qna, pred):
        return {
            "category": qna.get("category"),
            "subcategory": qna.get("subcategory"),
            "level": qna.get("level"),
            "answer": qna["answer"],
            "pred": pred[0],
            "response": pred[1],
        }
    
    def _make_failed(self, qna, error):
        return {
            "category": qna.get("category"),
            "subcategory": qna.get("subcategory"),
            "level": qna.get("level"),
            "answer": qna["answer"],
            "pred": "FAILED",
            "response": error,
        }
