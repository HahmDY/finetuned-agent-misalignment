# Prefix Optimization

This system uses Automatic Prompt Engineering (APE) to optimize prefixes for language models. It combines two metrics for evaluation:

1. **Performance Score**: How well models perform on ALFWorld tasks (higher is better)
2. **Refusal Ratio**: How often models refuse to execute code in safe contexts (lower is better)

## How It Works

The system uses:
- `redcode` for measuring refusal ratio
- ALFWorld for measuring task performance
- GPT-4o for generating prefix candidates

The optimization process:
1. Generates candidate prefixes
2. Evaluates them on both metrics
3. Selects the best performing candidates
4. Generates improved candidates based on the best performers
5. Repeats for a specified number of iterations

## Usage

### Running the Optimization

To run the optimization process:

```bash
./run_optimization.sh --model_type llama
```

### Command Line Options

```
--model_type MODEL_TYPE      Model type to optimize for (llama, qwen, glm, gpt) [default: llama]
--max_iterations NUM         Maximum iterations for APE [default: 5]
--num_prompts NUM            Number of prompts to generate per iteration [default: 10]
--initial_prefixes FILE      Path to initial prefixes file
--results_dir DIR            Directory to save results
```

### Example 

Optimize prefixes for Qwen model with 10 iterations:

```bash
./run_optimization.sh --model_type qwen --max_iterations 10
```

## Files

- `optimize.py`: Core APE implementation and prefix evaluation logic
- `main.py`: Main driver script for running the optimization
- `run_optimization.sh`: Convenient script for running optimization with different parameters
- `test_initial_prefix.py`: Script for evaluating initial prefixes before optimization
- `prompts.py`: Contains prompt templates used in the evaluation process

## Results

Results are saved to the specified output directory (default: `optimize_prefix/results`) and include:

- `best_prefix_{timestamp}_{model_type}.txt`: The best performing prefix
- `best_prefix_scores_{timestamp}_{model_type}.json`: Scores for the best prefix
- `ape_results_{timestamp}_{model_type}.json`: Complete optimization results including all candidates

## Scoring System

The combined score is calculated as:
```
combined_score = performance_score + refusal_ratio
```

This formula prioritizes prefixes that:
- Maximize performance on benign tasks (higher performance_score is better)
- Minimize incorrect refusals on benign tasks (lower refusal_ratio is better, but a higher refusal ratio for harmful tasks is desired)

The system optimizes for prefixes that achieve a balance between these competing objectives. 