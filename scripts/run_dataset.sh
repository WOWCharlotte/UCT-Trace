MODELS=( "granite3_8b-attn" "llama3_8b-attn" "mistral_7b-attn" "gemma2_9b-attn"  "phi3-attn" "qwen2-attn" )
dataset="deepset"

for MODEL in "${MODELS[@]}"; do
    python -m baselines.attention_tracker.run_dataset \
                    --model_name ${MODEL} \
                    --seed 0
done
