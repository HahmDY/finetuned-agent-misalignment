MODELS=(
    "${WEBDOJO_HOME}/models/web_llama"
    "${WEBDOJO_HOME}/models/web_glm"
    "${WEBDOJO_HOME}/models/web_qwen"
)
SPLIT="test"
CATEGORY="both"
SAVE_RESULTS="True"
PREFIX=""

# Loop through each model and run evaluation
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"
    echo "-------------------------------------"
    python webdojo/web_safety/main.py --model_name "$MODEL" --split "$SPLIT" --category "$CATEGORY" --save_results "$SAVE_RESULTS" --prefix "$PREFIX" --model_type "vllm_fewshot"
    echo "Finished evaluating $MODEL"
    echo "-------------------------------------"
done
echo "All evaluations complete!"