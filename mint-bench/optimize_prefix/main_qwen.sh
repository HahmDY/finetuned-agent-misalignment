SPLIT="test"
python optimize_prefix/main.py \
    --model_type "qwen" \
    --split "$SPLIT" \
    --config_path "configs/alfworld_qwen.json" \
    --v2 \
    --top_k 3 \
    --save_results