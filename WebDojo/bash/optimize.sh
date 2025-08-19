unset SSL_CERT_FILE
# Set the models to use for evaluation
EVAL_MODELS=("${WEBDOJO_HOME}/models/web_llama" "${WEBDOJO_HOME}/models/web_glm" "${WEBDOJO_HOME}/models/web_qwen")

# Set other parameters
CATEGORY="both"
SPLIT="validation"
NUM_EVAL_TASKS=14
RESULTS_DIR="/{WEBDOJO_HOME}/webdojo/optimize_prefix/results/"

# Run the evaluation for each model in EVAL_MODELS
for EVAL_MODEL in "${EVAL_MODELS[@]}"; do
    python webdojo/optimize_prefix/main.py \
        --eval_model "$EVAL_MODEL" \
        --category "$CATEGORY" \
        --split "$SPLIT" \
        --num_eval_tasks "$NUM_EVAL_TASKS" \
        --top_k 3 \
        --max_iterations 20 \
        --num_prompts 5 \
        --results_dir "$RESULTS_DIR" \
        --cache_dir "$RESULTS_DIR" \
        --save_results
done