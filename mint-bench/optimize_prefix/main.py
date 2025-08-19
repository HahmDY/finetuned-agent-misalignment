#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run the Automatic Prompt Engineer (APE) algorithm for optimizing prefixes
for model responses using redcode evaluation.
"""

import os
import argparse
import json
import sys
import time
import gc
import torch
import subprocess
from tqdm import tqdm

from optimize import APE, evaluate_prefix

# Set ALFWORLD_DATA environment variable
os.environ["ALFWORLD_DATA"] = os.path.abspath("data/processed/alfworld")
print(f"Set ALFWORLD_DATA to: {os.environ['ALFWORLD_DATA']}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the APE algorithm for optimizing prefixes")
    
    # Model settings
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model to use for generating prefixes (typically gpt-4o)")
    parser.add_argument("--model_type", type=str, default="llama", choices=["llama", "qwen", "gemma", "glm", "gpt", "gemini_ft", "gemini", "gpt_ft"],
                        help="Type of model to evaluate prefixes on")
    
    # APE settings
    parser.add_argument("--num_prompts", type=int, default=5,
                        help="Number of prompts to generate per iteration")
    parser.add_argument("--max_iterations", type=int, default=21,
                        help="Maximum number of iterations")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of top prompts to keep")
    parser.add_argument("--bottom_k", type=int, default=3,
                        help="Number of bottom prompts to keep")
    parser.add_argument("--only_top_k", action="store_true", default=False,
                        help="Whether to only keep top k prompts")
    parser.add_argument("--v2", action="store_true", default=False,
                        help="Whether to use v2 of the APE algorithm")
    
    # Evaluation settings
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"],
                        help="Split of tasks to evaluate")
    parser.add_argument("--config_path", type=str, 
                        default="configs/alfworld.json",
                        help="Path to alfworld config file for performance evaluation")
    
    # Initial prefixes
    parser.add_argument("--initial_prefixes_file", type=str, 
                        default="",
                        help="Path to a JSON file with initial prefixes")
    
    # Output settings
    parser.add_argument("--save_results", action="store_true", default=True,
                        help="Whether to save results")
    parser.add_argument("--results_dir", type=str, default="optimize_prefix/results_aaai",
                        help="Directory to save results")
    
    return parser.parse_args()

def load_initial_prefixes(file_path):
    """
    Load initial prefixes from a JSON file
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of prefix data dictionaries containing 'prefix', 'perform_score', and 'refusal_ratio'
    """
    if not file_path or not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        prefixes = []
        # Make sure the data is in the expected format
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "prefix" in item and "perform_score" in item and "refuse_score" in item:
                    prefixes.append(item)
        
        print(f"Loaded {len(prefixes)} initial prefixes")
        return prefixes
    
    except Exception as e:
        print(f"Error loading initial prefixes: {e}")
        return []

def run_alfworld_test(prefix, config_path, model_type, split="test"):
    """
    Run alfworld test with the given prefix and return success ratio
    
    Args:
        prefix: Prefix to test
        config_path: Path to alfworld config
        model_type: Model type (llama, qwen, glm)
    
    Returns:
        Performance score (success ratio)
    """
    try:
        # Create temporary output directory for this test
        output_dir = f"data/optimize_outputs/{model_type}/{int(time.time())}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Write the prefix to a temporary file
        prefix_file = os.path.join(output_dir, "prefix.txt")
        with open(prefix_file, "w") as f:
            f.write(prefix)
        # Create a copy of the config and modify the output dir
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update the output directory
        config['output_dir'] = output_dir
        
        # Write the updated config to a temporary file
        temp_config_path = os.path.join(output_dir, "temp_config.json")
        print(f"Writing updated config to {temp_config_path}")
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Use the temporary config path for the command
        config_path = temp_config_path
        
        # Run the test using the logic from test_initial_prefix.py
        main_script_path = os.path.abspath("mint/main.py")
        cmd = [
            "python", "-u", main_script_path, 
            f"--exp_config={config_path}", 
            f"--prefix={prefix}",
            f"--split={split}"
        ]
        
        print(f"Running alfworld test with command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Calculate success ratio
        results_path = os.path.join(output_dir, "results.jsonl")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = [json.loads(line) for line in f]
            
            success_count = 0
            total_count = len(results)
            
            for result in results:
                if result.get('state', {}).get('success', False):
                    success_count += 1
            
            success_ratio = success_count / total_count if total_count > 0 else 0
            print(f"Alfworld success ratio: {success_ratio:.4f}")
            return success_ratio
        else:
            print("Warning: Results file not found")
            return 0.0
    except Exception as e:
        print(f"Error running alfworld test: {e}")
        return 0.0

def eval_function(prefix, model_type, split="test", config_path=None):
    """
    Evaluation function for prefixes using both redcode refusal ratio and alfworld performance
    
    Args:
        prefix: Prefix to evaluate
        model_type: Model type to evaluate on
        split: Dataset split to use
        config_path: Path to alfworld config file
    
    Returns:
        Dictionary with evaluation scores
    """
    # Get refusal ratio from redcode
    print(f"Evaluating prefix with redcode: {prefix}")
    # Get the OpenAI API base URL from the config
    with open(config_path, 'r') as f:
        config = json.load(f)
    openai_api_base = config.get('agent', {}).get('config', {}).get('openai.api_base', "")
    
    redcode_results = evaluate_prefix(prefix, model_type=model_type, split=split, openai_base_url=openai_api_base)
    refuse_score = redcode_results["refuse_score"]
    
    # If config_path is provided, run alfworld test to get performance score
    if config_path:
        print(f"Running alfworld performance evaluation...")
        perform_score = run_alfworld_test(prefix, config_path, model_type, split)
    else:
        # Use the placeholder random score from evaluate_prefix if no config provided
        perform_score = redcode_results["perform_score"]
    
    # Calculate combined score (higher is better)
    # We want high perform_score and low refusal_ratio
    combined_score = perform_score + refuse_score
    
    return {
        'combined_score': combined_score,
        'perform_score': perform_score,
        'refuse_score': refuse_score
    }

def main():
    """Main function"""
    args = parse_args()
    
    # set the config path for each model_type
    if args.model_type == "llama":
        args.config_path = "configs/alfworld_llama.json"
    elif args.model_type == "qwen":
        args.config_path = "configs/alfworld_qwen.json"
    elif args.model_type == "gemma":
        args.config_path = "configs/alfworld_gemma.json"
    elif args.model_type == "glm":
        args.config_path = "configs/alfworld_glm.json"
    elif args.model_type == "gpt":
        args.config_path = "configs/alfworld_gpt.json"
    elif args.model_type == "gpt_ft":
        args.config_path = "configs/alfworld_gpt_ft.json"
    elif args.model_type == "gemini_ft":
        args.config_path = "configs/alfworld_gemini_ft.json"
    elif args.model_type == "gemini":
        args.config_path = "configs/alfworld_gemini.json"

    print(f"Using config path: {args.config_path} for {args.model_type} model")
    # Create directory for results if it doesn't exist
    if args.save_results and not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    # Load initial prefixes if provided
    initial_prefixes = load_initial_prefixes(args.initial_prefixes_file)
    
    # Initialize the APE algorithm
    ape = APE(
        eval_function=lambda prefix, model_type: eval_function(
            prefix, model_type, args.split, args.config_path
        ),
        initial_prefixes=initial_prefixes,
        model=args.model,
        num_prompts=args.num_prompts,
        max_iterations=args.max_iterations,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        only_top_k=args.only_top_k,
        save_results=args.save_results,
        results_dir=args.results_dir,
        model_type=args.model_type,
        v2=args.v2
    )
    
    # Run the APE algorithm
    print("Starting APE algorithm...")
    
    # Force garbage collection before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    best_prefix, best_score_dict = ape.run()
    
    print("\n" + "="*50)
    print("APE algorithm completed!")
    print("="*50)
    print(f"Best prefix: {best_prefix}")
    print(f"Performance score: {best_score_dict.get('perform_score', 0.0):.4f}")
    print(f"Refusal ratio: {best_score_dict.get('refuse_score', 0.0):.4f}")
    print(f"Combined score: {best_score_dict.get('combined_score', 0.0):.4f}")
    
    # Save the best prefix to a file
    if args.save_results:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save best prefix to text file
        best_prefix_file = os.path.join(args.results_dir, f"best_prefix_{timestamp}_{args.model_type}.txt")
        with open(best_prefix_file, "w") as f:
            f.write(best_prefix)
        print(f"Best prefix saved to {best_prefix_file}")
        
        # Save best prefix scores to JSON file
        best_score_file = os.path.join(args.results_dir, f"best_prefix_scores_{timestamp}_{args.model_type}.json")
        with open(best_score_file, "w") as f:
            json.dump(best_score_dict, f, indent=2)
        print(f"Best prefix scores saved to {best_score_file}")
        
        # Save full APE results
        ape_results_file = os.path.join(args.results_dir, f"ape_results_{timestamp}_{args.model_type}.json")
        with open(ape_results_file, "w") as f:
            # Create a results dictionary with all relevant information
            results = {
                "best_prefix": best_prefix,
                "best_scores": best_score_dict,
                "model_type": args.model_type,
                "split": args.split,
                "timestamp": timestamp
            }
            json.dump(results, f, indent=2)
        print(f"Full APE results saved to {ape_results_file}")

if __name__ == "__main__":
    main() 