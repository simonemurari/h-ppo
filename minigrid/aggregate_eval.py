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
import ppo_eval_check_values
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


def _normalize_eval_episode_suffix(model_dir_name: str, eval_episodes: int) -> str:
    """Return model filename stem with canonical _<N>_eval_eps suffix."""
    suffix = f"_{int(eval_episodes)}_eval_eps"
    if model_dir_name.endswith(suffix):
        return model_dir_name

    # Handle legacy/manual suffixes like *_20_eps.
    legacy_suffix = f"_{int(eval_episodes)}_eps"
    if model_dir_name.endswith(legacy_suffix):
        model_dir_name = model_dir_name[: -len(legacy_suffix)]

    return f"{model_dir_name}{suffix}"


def _append_eval_code_suffix(name_stem: str, eval_code: str) -> str:
    """Append eval_code to filename stem when provided and not already present."""
    if not eval_code:
        return name_stem
    if name_stem.endswith(eval_code):
        return name_stem
    return f"{name_stem}{eval_code}"


def _safe_float(value, default=None):
    """Best-effort float conversion; return default on missing/invalid input."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=None):
    """Best-effort int conversion; return default on missing/invalid input."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_prob(probs, idx):
    """Safely extract probability idx from a probs-like sequence."""
    if probs is None:
        return None
    try:
        if len(probs) <= idx:
            return None
        return _safe_float(probs[idx], default=None)
    except (TypeError, ValueError, IndexError):
        return None

def main():
    args = tyro.cli(Args)
    
    # 5 different master seeds for the 5 evaluation runs
    master_seeds = args.master_seeds
    print(f"Using Master Seeds: {master_seeds}")
    all_returns = []
    all_step_rows = []
    all_episode_summary_rows = []
    per_seed_step_rows = {}
    per_seed_episode_summary_rows = {}
    
    # Decide which evaluation function to use based on exp_name or other logic
    if "random_rules" in args.eval_type:
        eval_func = ppo_eval_random_rules.evaluate
        eval_type_subfolder = "random_rules"
    elif "check_values" in args.eval_type:
        eval_func = ppo_eval_check_values.evaluate
        eval_type_subfolder = "check_values"
    elif "h_ppo" in args.eval_type or args.epsilon is not None:
        eval_func = h_ppo_eval.evaluate
        eval_type_subfolder = f"h_ppo_eps_{args.epsilon}"
    elif "standard" in args.eval_type:
        eval_func = ppo_eval.evaluate
        eval_type_subfolder = "standard"
    else:
        raise ValueError(f"Unknown eval_type: {args.eval_type}, must be one of 'standard', 'random_rules', 'check_values', or 'h_ppo'.")
    
    # Print evaluation info
    eval_info = f"eval_type={args.eval_type}"
    if args.epsilon is not None:
        eval_info += f", epsilon={args.epsilon}"
    print(f"Starting aggregate evaluation on {args.env_id} ({eval_info})")
    print(f"{len(master_seeds)} seeds, {args.eval_episodes} episodes each. Total: {len(master_seeds) * args.eval_episodes} episodes.")

    # For full evaluations, keep return-only flow and skip detailed per-step metrics.
    low_eval = args.eval_episodes < 1000

    collect_episode_metrics = low_eval and args.model_name == "ppo" and eval_func in {
        ppo_eval.evaluate,
        ppo_eval_check_values.evaluate,
    }

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

        if eval_func == ppo_eval_check_values.evaluate:
            eval_kwargs["rolling_window_size"] = args.rolling_window_size
            eval_kwargs["value_skip_threshold"] = args.value_skip_threshold

        if collect_episode_metrics:
            returns, metrics = eval_func(**eval_kwargs, return_episode_metrics=True)
        else:
            returns = eval_func(**eval_kwargs)
            metrics = None
        all_returns.extend(returns)

        if metrics is not None:
            per_step = metrics.get("per_step", [])
            if s not in per_seed_step_rows:
                per_seed_step_rows[s] = []
            for row in per_step:
                ep_idx = _safe_int(row.get("Episode"))
                if ep_idx is None or ep_idx < 1 or ep_idx > len(returns):
                    continue
                episode_return = returns[ep_idx - 1]
                probs = row.get("Probs")
                step_row = {
                    "GlobalStep": len(all_step_rows) + 1,
                    "MasterSeed": s,
                    "Episode": ep_idx,
                    "EpisodeStep": _safe_int(row.get("Step")),
                    "Return": float(episode_return.item() if hasattr(episode_return, "item") else episode_return),
                    "Action": _safe_int(row.get("Action")),
                    "LogProb": _safe_float(row.get("LogProb")),
                    "Entropy": _safe_float(row.get("Entropy")),
                    "Value": _safe_float(row.get("Value")),
                    "Prob0": _safe_prob(probs, 0),
                    "Prob1": _safe_prob(probs, 1),
                    "Prob2": _safe_prob(probs, 2),
                    "Prob3": _safe_prob(probs, 3),
                    "Prob4": _safe_prob(probs, 4),
                    "Prob5": _safe_prob(probs, 5),
                    "Prob6": _safe_prob(probs, 6),
                }
                all_step_rows.append(step_row)
                per_seed_step_rows[s].append(step_row)

            per_episode = metrics.get("per_episode", {})
            if s not in per_seed_episode_summary_rows:
                per_seed_episode_summary_rows[s] = []
            for ep_idx, ret in enumerate(returns, start=1):
                episode_row = {
                    "GlobalEpisode": i * args.eval_episodes + ep_idx,
                    "MasterSeed": s,
                    "Episode": ep_idx,
                    "Return": float(ret.item() if hasattr(ret, "item") else ret),
                    "ActionMean": _safe_float((per_episode.get("action_mean") or [None] * len(returns))[ep_idx - 1]),
                    "ActionStd": _safe_float((per_episode.get("action_std") or [None] * len(returns))[ep_idx - 1]),
                    "LogProbMean": _safe_float((per_episode.get("logprob_mean") or [None] * len(returns))[ep_idx - 1]),
                    "LogProbStd": _safe_float((per_episode.get("logprob_std") or [None] * len(returns))[ep_idx - 1]),
                    "EntropyMean": _safe_float((per_episode.get("entropy_mean") or [None] * len(returns))[ep_idx - 1]),
                    "EntropyStd": _safe_float((per_episode.get("entropy_std") or [None] * len(returns))[ep_idx - 1]),
                    "ValueMean": _safe_float((per_episode.get("value_mean") or [None] * len(returns))[ep_idx - 1]),
                    "ValueStd": _safe_float((per_episode.get("value_std") or [None] * len(returns))[ep_idx - 1]),
                }
                all_episode_summary_rows.append(episode_row)
                per_seed_episode_summary_rows[s].append(episode_row)

    # Calculate statistics
    all_returns = np.array(all_returns)
    mean_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    
    print("\nFinal Results:")
    print(f"Mean Return: {mean_return:.2f}")
    print(f"Std Return: {std_return:.2f}")

    # Save to CSV with split folder structure:
    # - eval_episodes >= 1000: aggregate_evals/<env>/<eval_type>/<model>.csv  (Return-only schema)
    # - eval_episodes  < 1000: aggregate_evals_logs/<env>/<eval_type>/<model>.csv  (detailed schema)
    base_dir = "aggregate_evals_logs" if low_eval else "aggregate_evals"
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

    model_file_stem = _normalize_eval_episode_suffix(model_dir_name, args.eval_episodes)
    # Append check_values with its config params so different window/threshold runs are distinct.
    if eval_type_subfolder == "check_values":
        cv_suffix = f"_check_values_w{args.rolling_window_size}_t{args.value_skip_threshold}"
        if not model_file_stem.endswith(cv_suffix):
            model_file_stem = f"{model_file_stem}{cv_suffix}"
    model_file_stem = _append_eval_code_suffix(model_file_stem, args.eval_code)
    csv_path = os.path.join(target_dir, f"{model_file_stem}.csv")

    print(f"Saving returns to {csv_path}")
    if low_eval:
        # Detailed per-step schema for low-eval runs (< 1000 episodes).
        fieldnames = ["GlobalStep", "MasterSeed", "Episode", "EpisodeStep",
                      "Return", "Action", "LogProb", "Entropy", "Value",
                      "Prob0", "Prob1", "Prob2", "Prob3", "Prob4", "Prob5", "Prob6"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_step_rows)
    else:
        # Lean Return-only schema for full-eval runs (>= 1000 episodes).
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Return"])
            for r in all_returns:
                val = r.item() if hasattr(r, "item") else float(r)
                writer.writerow([val])

    # Track one W&B run per master seed (same group).
    if args.track:
        import wandb

        total_logged_points = 0
        for s in master_seeds:
            seed_step_rows = per_seed_step_rows.get(s, [])
            seed_episode_rows = per_seed_episode_summary_rows.get(s, [])
            if not seed_step_rows and not seed_episode_rows:
                continue

            run = wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                group=args.group_name,
                name=f"aggregate_eval_seed_{s}_{model_dir_name}_{args.eval_type}_{args.env_id}_{args.n_keys}_{int(time.time())}",
                config=vars(args),
                save_code=True,
                job_type="aggregate_eval_seed",
                reinit=True,
            )

            wandb.define_metric("aggregate_eval/global_step")
            wandb.define_metric("aggregate_eval/*", step_metric="aggregate_eval/global_step")
            wandb.define_metric("aggregate_eval_episode/episode")
            wandb.define_metric("aggregate_eval_episode/*", step_metric="aggregate_eval_episode/episode")

            for local_step, row in enumerate(seed_step_rows, start=1):
                payload = {
                    "aggregate_eval/global_step": local_step,
                    "aggregate_eval/master_seed": row["MasterSeed"],
                    "aggregate_eval/episode": row["Episode"],
                    "aggregate_eval/episode_step": row["EpisodeStep"],
                    "aggregate_eval/return": row["Return"],
                    "aggregate_eval/action": row["Action"],
                    "aggregate_eval/logprob": row["LogProb"],
                    "aggregate_eval/entropy": row["Entropy"],
                    "aggregate_eval/value": row["Value"],
                }
                wandb.log(payload)

            for row in seed_episode_rows:
                wandb.log(
                    {
                        "aggregate_eval_episode/episode": row["Episode"],
                        "aggregate_eval_episode/master_seed": row["MasterSeed"],
                        "aggregate_eval_episode/return": row["Return"],
                        "aggregate_eval_episode/action_mean": row["ActionMean"],
                        "aggregate_eval_episode/action_std": row["ActionStd"],
                        "aggregate_eval_episode/logprob_mean": row["LogProbMean"],
                        "aggregate_eval_episode/logprob_std": row["LogProbStd"],
                        "aggregate_eval_episode/entropy_mean": row["EntropyMean"],
                        "aggregate_eval_episode/entropy_std": row["EntropyStd"],
                        "aggregate_eval_episode/value_mean": row["ValueMean"],
                        "aggregate_eval_episode/value_std": row["ValueStd"],
                    }
                )

            seed_returns = [r["Return"] for r in seed_episode_rows]
            if seed_returns:
                wandb.summary["aggregate_eval/mean_return"] = float(np.mean(seed_returns))
                wandb.summary["aggregate_eval/std_return"] = float(np.std(seed_returns))
            wandb.summary["aggregate_eval/num_points"] = len(seed_step_rows)
            run.finish()
            total_logged_points += len(seed_step_rows)

        print(
            f"Logged per-seed aggregate evaluation runs to Weights & Biases ({len(master_seeds)} seeds, {total_logged_points} step points)."
        )
    elif not args.track:
        print("Skipping W&B logging because track=False in config_eval.")

if __name__ == "__main__":
    main()
