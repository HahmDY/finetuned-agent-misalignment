unset SSL_CERT_FILE
SPLIT="test"
python optimize_prefix/main.py \
    --model_type "gpt" \
    --split "$SPLIT" \
    --config_path "configs/alfworld_gpt.json" \
    --save_results \
    --v2 \
    --top_k 3 \
