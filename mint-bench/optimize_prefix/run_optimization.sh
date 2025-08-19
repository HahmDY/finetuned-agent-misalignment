#!/bin/bash

# Make script exit on first error
set -e

# Default parameters
MODEL_TYPE="llama"
MAX_ITERATIONS=5
NUM_PROMPTS=10
CONFIG_PATH="${PING_HOME}/mint-bench/configs/alfworld.json"
INITIAL_PREFIXES_FILE="${PING_HOME}/mint-bench/optimize_prefix/initial_prefixes_evaluated_llama.json"
RESULTS_DIR="${PING_HOME}/mint-bench/optimize_prefix/results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --max_iterations)
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --num_prompts)
      NUM_PROMPTS="$2"
      shift 2
      ;;
    --initial_prefixes)
      INITIAL_PREFIXES_FILE="$2"
      shift 2
      ;;
    --results_dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --model_type MODEL_TYPE      Model type to optimize for (llama, qwen, glm) [default: llama]"
      echo "  --max_iterations NUM         Maximum iterations for APE [default: 5]"
      echo "  --num_prompts NUM            Number of prompts to generate per iteration [default: 20]"
      echo "  --initial_prefixes FILE      Path to initial prefixes file"
      echo "  --results_dir DIR            Directory to save results"
      echo "  --help                       Display this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Make sure the results directory exists
mkdir -p "$RESULTS_DIR"

echo "=== Starting Prefix Optimization ==="
echo "Model type: $MODEL_TYPE"
echo "Max iterations: $MAX_ITERATIONS"
echo "Num prompts: $NUM_PROMPTS"
echo "Initial prefixes file: $INITIAL_PREFIXES_FILE"
echo "Results directory: $RESULTS_DIR"

# Run optimization
python -u optimize_prefix/main.py \
  --model_type "$MODEL_TYPE" \
  --max_iterations "$MAX_ITERATIONS" \
  --num_prompts "$NUM_PROMPTS" \
  --initial_prefixes_file "$INITIAL_PREFIXES_FILE" \
  --results_dir "$RESULTS_DIR" \
  --config_path "$CONFIG_PATH"

echo "=== Optimization Complete ==="
echo "Results saved to $RESULTS_DIR" 