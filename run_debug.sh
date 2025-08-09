#!/bin/bash

### Parallel execution version of run_all.sh
env_files=(.env_gpt5-mini) 
is_debug=False
batch_size=30
max_tokens=256
temperature=0.01
max_parallel_jobs=3

echo "Found the following .env files:"
for env_file in "${env_files[@]}"; do
    echo "$env_file"
done

# 함수: 단일 모델의 모든 벤치마크 실행
run_model() {
    local env_file=$1
    local model_provider=$2
    
    echo "Starting evaluation for $env_file"
    
    # CLIcK
    # DOTENV_PATH="$env_file" python click_main.py \
    #     --is_debug "$is_debug" \
    #     --model_provider "$model_provider" \
    #     --batch_size "$batch_size" \
    #     --max_tokens "$max_tokens" \
    #     --temperature "$temperature" \
    #     --template_type chat &
    
    # HAERAE 1.0
    # DOTENV_PATH="$env_file" python haerae_main.py \
    #     --is_debug "$is_debug" \
    #     --model_provider "$model_provider" \
    #     --batch_size "$batch_size" \
    #     --max_tokens "$max_tokens" \
    #     --temperature "$temperature" \
    #     --template_type chat &
    
    # KMMLU
    DOTENV_PATH="$env_file" python kmmlu_main.py \
        --is_debug "$is_debug" \
        --model_provider "$model_provider" \
        --batch_size "$batch_size" \
        --max_tokens "$max_tokens" \
        --temperature "$temperature" \
        --template_type chat \
        --is_hard False \
        --use_few_shot False &
    
    # # KMMLU (HARD)
    # DOTENV_PATH="$env_file" python kmmlu_main.py \
    #     --is_debug "$is_debug" \
    #     --model_provider "$model_provider" \
    #     --batch_size "$batch_size" \
    #     --max_tokens "$max_tokens" \
    #     --temperature "$temperature" \
    #     --template_type chat \
    #     --is_hard True \
    #     --use_few_shot False &
    
    wait  # 해당 모델의 모든 작업이 완료될 때까지 대기
    echo "Completed evaluation for $env_file"
}

# 병렬 실행 관리
job_count=0
for env_file in "${env_files[@]}"; do
    if [[ "$env_file" == .env_gpt* ]]; then
        model_provider="azureopenai"
    else
        model_provider="azureml"
    fi
    
    # 백그라운드에서 실행
    run_model "$env_file" "$model_provider" &
    
    ((job_count++))
    
    # 최대 병렬 작업 수에 도달하면 일부 작업이 완료될 때까지 대기
    if (( job_count >= max_parallel_jobs )); then
        wait -n  # 하나의 작업이 완료될 때까지 대기
        ((job_count--))
    fi
done

wait  # 모든 작업 완료 대기
echo "All evaluations completed!"