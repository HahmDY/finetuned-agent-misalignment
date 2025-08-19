unset SSL_CERT_FILE
SPLIT="test"
python optimize_prefix/main.py \
    --model_type "llama" \
    --split "$SPLIT" \
    --config_path "configs/alfworld_llama.json" \
    --v2 \
    --top_k 3 \
    --save_results