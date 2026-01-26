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
import ppo
import h_ppo_product

def main():
    args = tyro.cli(Args)
    
    # 5 different master seeds for the 5 evaluation runs
    master_seeds = args.master_seeds
    print(f"Using Master Seeds: {master_seeds}")
    all_returns = []
    
    # Decide which evaluation function to use based on exp_name or other logic
    if "random_rules" in args.eval_type:
        eval_func = ppo_eval_random_rules.evaluate
        eval_type_subfolder = "random_rules"
    elif "h_ppo" in args.eval_type or args.epsilon is not None:
        # h_ppo eval_type only works with h_ppo_product model
        if args.model_name != "h_ppo_product":
            raise ValueError(
                f"h_ppo eval_type can only be used with 'h_ppo_product' model, "
                f"but got '{args.model_name}'. Use 'standard' or 'random_rules' eval_type instead."
            )
        eval_func = h_ppo_eval.evaluate
        eval_type_subfolder = f"h_ppo_eps_{args.epsilon}"
    elif "standard" in args.eval_type:
        eval_func = ppo_eval.evaluate
        eval_type_subfolder = "standard"
    else:
        raise ValueError(f"Unknown eval_type: {args.eval_type}, must be one of 'standard', 'random_rules', or 'h_ppo'.")
    
    print(f"Starting aggregate evaluation for {args.exp_name} on {args.env_id}")
    print(f"{len(master_seeds)} seeds, {args.eval_episodes} episodes each. Total: {len(master_seeds) * args.eval_episodes} episodes.")

    for i, s in enumerate(master_seeds):
        print(f"\n--- Run {i+1}/{len(master_seeds)} with Master Seed {s} ---")
        args.seed = s  # Update seed for each run
        
        if eval_func == h_ppo_eval.evaluate:
            model_class = h_ppo_product.Agent
        else:
            model_class = ppo.Agent

        eval_kwargs = {
            "model_path": args.model_path,
            "make_env": ppo.make_env,
            "env_id": args.env_id,
            "n_keys": args.n_keys,
            "eval_episodes": args.eval_episodes,
            "run_name": f"{args.env_id}_seed_{s}",
            "seed": s,
            "group_name": args.group_name,
            "Model": model_class,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "capture_video": False,
            "track": False, # Don't log individual runs
            "from_config": False,
            "random_color": args.random_color
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

    # Save to CSV with new folder structure:
    # aggregate_evals/<env_id>_<n_keys>keys/<eval_type_subfolder>/<model_name>.csv
    base_dir = "aggregate_evals"
    env_sub_dir = f"{args.env_id}_{args.n_keys}keys"
    
    
    target_dir = os.path.join(base_dir, env_sub_dir, eval_type_subfolder)
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
            # Extract scalar from numpy array using .item()
            val = r.item() if hasattr(r, 'item') else float(r)
            writer.writerow([val])

if __name__ == "__main__":
    main()
