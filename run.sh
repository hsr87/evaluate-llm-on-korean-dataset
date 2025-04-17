# Please make sure to set the correct environment variables in the .env file before running this script
#
# If you are using Azure AI Foundry, you can set the following environment variables in the .env file:
# AZURE_AI_INFERENCE_KEY=<your-ai-inference-key>
# AZURE_AI_INFERENCE_ENDPOINT=https://<your-endpoint-name>.services.ai.azure.com/models
# AZURE_AI_DEPLOYMENT_NAME=Phi-4


### CLIcK
python click_main.py \
    --is_debug False \
    --model_provider azureopenai \
    --batch_size 10 \
    --max_tokens 256 \
    --temperature 0.01 \
    --template_type chat 

### HAERAE 1.0
python haerae_main.py \
    --is_debug False \
    --model_provider azureopenai \
    --batch_size 10 \
    --max_tokens 256 \
    --temperature 0.01 \
    --template_type chat 

## KMMLU
python kmmlu_main.py \
    --is_debug False \
    --model_provider azureopenai \
    --batch_size 10 \
    --max_tokens 256 \
    --temperature 0.01 \
    --template_type chat \
    --is_hard False \
    --use_few_shot False