import numpy as np
import torch
import tyro
import os
import csv
import time
from config_eval import Args

# Import the specific scripts to get their evaluate functions
import ppo_eval
import h_ppo_eval
import ppo_eval_random_rules
from ppo import make_env, Agent as PPOAgent
from h_ppo_PSM import Agent as HPPOPSMAgent
from h_ppo_product import Agent as HPPOProductAgent
from h_ppo_PSM_netembed import Agent as HPPOPSMnetembedAgent

def get_agent_class(model_name: str):
    """Return the appropriate Agent class based on model name."""
    model_name = model_name.lower()
    if model_name == "h_ppo_psm":
        return HPPOPSMAgent
    elif model_name == "h_ppo_product":
        return HPPOProductAgent
    elif model_name == "h_ppo_psm_netembed":
        return HPPOPSMnetembedAgent
    else:
        return PPOAgent

def main():
    args = tyro.cli(Args)
    
    # 5 different master seeds for the 5 evaluation runs
    master_seeds = args.master_seeds
    print(f"Using Master Seeds: {master_seeds}")
    all_returns = []
    all_episode_rows = []
    
    # Decide which evaluation function to use based on exp_name or other logic
    if "random_rules" in args.eval_type:
        eval_func = ppo_eval_random_rules.evaluate
        eval_type_subfolder = "random_rules"
    elif "h_ppo" in args.eval_type or args.epsilon is not None:
        eval_func = h_ppo_eval.evaluate
        eval_type_subfolder = f"h_ppo_eps_{args.epsilon}"
    elif "standard" in args.eval_type:
        eval_func = ppo_eval.evaluate
        eval_type_subfolder = "standard"
    else:
        raise ValueError(f"Unknown eval_type: {args.eval_type}, must be one of 'standard', 'random_rules', or 'h_ppo'.")
    
    # Print evaluation info
    eval_info = f"eval_type={args.eval_type}"
    if args.epsilon is not None:
        eval_info += f", epsilon={args.epsilon}"
    print(f"Starting aggregate evaluation on {args.env_id} ({eval_info})")
    print(f"{len(master_seeds)} seeds, {args.eval_episodes} episodes each. Total: {len(master_seeds) * args.eval_episodes} episodes.")

    collect_episode_metrics = args.model_name == "ppo" and eval_func == ppo_eval.evaluate

    for i, s in enumerate(master_seeds):
        print(f"\n--- Run {i+1}/{len(master_seeds)} with Master Seed {s} ---")
        args.seed = s  # Update seed for each run

        eval_kwargs = {
            "model_path": args.model_path,
            "make_env": make_env,
            "env_id": args.env_id,
            "n_keys": args.n_keys,
            "eval_episodes": args.eval_episodes,
            "run_name": f"{args.env_id}_seed_{s}",
            "seed": s,
            "group_name": args.group_name,
            "Model": get_agent_class(args.model_name),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "capture_video": False,
            "track": False, # Don't log individual runs
            "from_config": False,
            "random_color": args.random_color
        }

        if eval_func == h_ppo_eval.evaluate:
            eval_kwargs["epsilon"] = args.epsilon

        if collect_episode_metrics:
            returns, metrics = eval_func(**eval_kwargs, return_episode_metrics=True)
        else:
            returns = eval_func(**eval_kwargs)
            metrics = None
        all_returns.extend(returns)

        for ep_idx, ret in enumerate(returns, start=1):
            step_idx = i * args.eval_episodes + ep_idx
            row = {
                "Step": step_idx,
                "MasterSeed": s,
                "Episode": ep_idx,
                "Return": float(ret.item() if hasattr(ret, "item") else ret),
            }
            if metrics is not None:
                row["Action"] = float(metrics["action"][ep_idx - 1])
                row["LogProb"] = float(metrics["logprob"][ep_idx - 1])
                row["Entropy"] = float(metrics["entropy"][ep_idx - 1])
                row["Value"] = float(metrics["value"][ep_idx - 1])
            all_episode_rows.append(row)

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
        if all_episode_rows:
            fieldnames = ["Step", "MasterSeed", "Episode", "Return"]
            if collect_episode_metrics:
                fieldnames.extend(["Action", "LogProb", "Entropy", "Value"])

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_episode_rows)
        else:
            # Backward-compatible schema when no per-episode rows were collected.
            writer = csv.writer(f)
            writer.writerow(["Return"])
            for r in all_returns:
                val = r.item() if hasattr(r, 'item') else float(r)
                writer.writerow([val])

    # Track only aggregate evaluation points (5 seeds x eval_episodes), never per-seed runs.
    if args.track and all_episode_rows:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.group_name,
            name=f"aggregate_eval_{model_dir_name}_{args.eval_type}_{args.env_id}_{int(time.time())}",
            config=vars(args),
            save_code=True,
            job_type="aggregate_eval",
            reinit=True,
        )

        wandb.define_metric("aggregate_eval/step")
        wandb.define_metric("aggregate_eval/*", step_metric="aggregate_eval/step")

        for row in all_episode_rows:
            payload = {
                "aggregate_eval/step": row["Step"],
                "aggregate_eval/master_seed": row["MasterSeed"],
                "aggregate_eval/episode": row["Episode"],
                "aggregate_eval/return": row["Return"],
            }
            if collect_episode_metrics:
                payload["aggregate_eval/action"] = row["Action"]
                payload["aggregate_eval/logprob"] = row["LogProb"]
                payload["aggregate_eval/entropy"] = row["Entropy"]
                payload["aggregate_eval/value"] = row["Value"]
            wandb.log(payload)

        wandb.summary["aggregate_eval/num_points"] = len(all_episode_rows)
        wandb.summary["aggregate_eval/mean_return"] = float(mean_return)
        wandb.summary["aggregate_eval/std_return"] = float(std_return)
        run.finish()
        print(f"Logged {len(all_episode_rows)} aggregate evaluation points to Weights & Biases.")
    elif not args.track:
        print("Skipping W&B logging because track=False in config_eval.")

if __name__ == "__main__":
    main()
