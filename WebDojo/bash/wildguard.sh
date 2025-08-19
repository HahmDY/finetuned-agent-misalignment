MODELS=(
    "${WEBDOJO_HOME}/models/web_qwen"
    "${WEBDOJO_HOME}/models/web_glm"
    "${WEBDOJO_HOME}/models/web_llama"
)
SPLIT="test"
CATEGORY="both"
SAVE_RESULTS="True"

# Run evaluation for each model
for MODEL in "${MODELS[@]}"; do
    python webdojo/guard/wildguard.py --model_name "$MODEL" --split "$SPLIT" --category "$CATEGORY" --save_results "$SAVE_RESULTS" --model_type "ft"
done