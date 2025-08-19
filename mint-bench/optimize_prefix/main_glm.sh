unset SSL_CERT_FILE
SPLIT="test"
python optimize_prefix/main.py \
    --model_type "glm" \
    --split "$SPLIT" \
    --config_path "configs/alfworld_glm.json" \
    --v2 \
    --top_k 3 \
    --save_results