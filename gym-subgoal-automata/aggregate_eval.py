import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="urllib3")

import numpy as np
import torch
import tyro
import os
import csv
from config_eval import Args

# Import the specific scripts to get their evaluate functions
import ppo_eval
import h_ppo_eval
import ppo_eval_random_rules
from ppo import make_env, Agent as PPOAgent
from h_ppo_product import Agent as HPPOProductAgent

def get_agent_class(model_name: str):
    """Return the appropriate Agent class based on model name."""
    if model_name == "h_ppo_product":
        return HPPOProductAgent
    else:
        return PPOAgent

def main():
    args = tyro.cli(Args)
    
    # 5 different master seeds for the 5 evaluation runs
    master_seeds = args.master_seeds
    print(f"Using Master Seeds: {master_seeds}")
    all_returns = []
    
    # Decide which evaluation function to use based on exp_name or other logic
    if "random_rules" in args.eval_type:
        eval_func = ppo_eval_random_rules.evaluate
    elif "h_ppo" in args.eval_type or args.epsilon is not None:
        eval_func = h_ppo_eval.evaluate
    elif "standard" in args.eval_type:
        eval_func = ppo_eval.evaluate
    else:
        raise ValueError(f"Unknown eval_type: {args.eval_type}, must be one of 'standard', 'random_rules', or 'h_ppo'.")
    
    print(f"Starting aggregate evaluation for {args.exp_name} on {args.env_id}")
    print(f"{len(master_seeds)} seeds, {args.eval_episodes} episodes each. Total: {len(master_seeds) * args.eval_episodes} episodes.")

    for i, s in enumerate(master_seeds):
        print(f"\n--- Run {i+1}/{len(master_seeds)} with Master Seed {s} ---")
        args.seed = s  # Update seed for each run
        
        eval_kwargs = {
            "model_path": args.model_path,
            "make_env": make_env,
            "env_id": args.env_id,
            "eval_episodes": args.eval_episodes,
            "run_name": f"{args.env_id}_seed_{s}",
            "seed": s,
            "group_name": args.group_name,
            "Model": get_agent_class(args.model_name),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "capture_video": False,
            "track": False, # Don't log individual runs
            "from_config": False,
        }

        if eval_func == h_ppo_eval.evaluate:
            eval_kwargs["epsilon"] = args.epsilon

        returns = eval_func(**eval_kwargs)
        all_returns.extend(returns)

    # Calculate statistics
    all_returns = np.array(all_returns)
    mean_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    
    print("\nFinal Results:")
    print(f"Mean Return: {mean_return:.2f}")
    print(f"Std Return: {std_return:.2f}")

    # Save to CSV
    # Create aggregate_evals folder with nested structure: task/eval_type/
    base_dir = "aggregate_evals"
    target_dir = os.path.join(base_dir, args.task, args.eval_type)
    os.makedirs(target_dir, exist_ok=True)

    # Extract filename from model_path (part between first and second /)
    # model_path example: models/h_ppo_symloss_theta_8x8_1keys_theta_0.5/h_ppo_...
    path_parts = args.model_path.split("/")
    if len(path_parts) > 1:
        model_dir_name = path_parts[1]
    else:
        model_dir_name = "unknown_model"
    
    csv_path = os.path.join(target_dir, f"{model_dir_name}.csv")
    
    print(f"Saving returns to {csv_path}")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Return"])
        for r in all_returns:
            writer.writerow([r])

if __name__ == "__main__":
    main()
