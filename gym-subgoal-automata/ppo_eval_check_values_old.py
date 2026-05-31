from typing import Callable
import tyro
import time
import gym
import wandb
import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from config_eval import Args
import warnings
warnings.filterwarnings("ignore")


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    seed: int,
    group_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    track: bool = False,
    from_config: bool = True,
    return_episode_metrics: bool = False,
    rolling_window_size: int = 3,
    value_skip_threshold: float = 0.0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    envs = gym.vector.SyncVectorEnv([make_env(env_id, seed, 0, capture_video, run_name)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    if from_config is True:
        args = tyro.cli(Args)
        env_id = args.env_id
        model_path = args.model_path
        eval_episodes = args.eval_episodes
        run_name = f"{args.task}__{args.exp_name}__{args.seed}__{int(time.time())}-eval"
        group_name = args.group_name
        capture_video = args.capture_video
        seed = args.seed
        track = args.track
        wandb_project_name = args.wandb_project_name
        wandb_entity = args.wandb_entity
        rolling_window_size = args.rolling_window_size
        value_skip_threshold = args.value_skip_threshold

    if track:
        run_config = vars(args) if from_config else {}

        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            sync_tensorboard=True,
            config=run_config,
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=group_name,
        )

        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in run_config.items()])),
        )

    seeds = random.sample(range(0, 10**9), eval_episodes)
    envs.seed(seeds[0])
    obs = envs.reset()

    episodic_returns = []
    episodic_actions_mean = []
    episodic_actions_std = []
    episodic_logprobs_mean = []
    episodic_logprobs_std = []
    episodic_entropies_mean = []
    episodic_entropies_std = []
    episodic_values_mean = []
    episodic_values_std = []
    per_step_metrics = []
    episode_actions = []
    episode_logprobs = []
    episode_entropies = []
    episode_values = []
    value_block = []
    action_block = []
    total_eval_steps = 0
    pbar = tqdm(total=eval_episodes)
    tqdm.write(f"Model path: {model_path}")
    tqdm.write(f"Evaluating {env_id} with {eval_episodes} episodes (check_values_OLD)")
    tqdm.write(f"  rolling_window_size={rolling_window_size}, value_skip_threshold={value_skip_threshold}")

    while len(episodic_returns) < eval_episodes:
        obs_t = torch.Tensor(obs).to(device)
        actions, logprobs, entropies, values = agent.get_action_and_value(obs_t)

        # Default: use PPO sampled action.
        chosen_action = int(actions.item())
        value_val = float(values.item())
        entropy_val = float(entropies.item())
        logprob_val = float(logprobs.item())

        value_block.append(value_val)
        action_block.append(chosen_action)

        current_episode_step = len(episode_actions) + 1
        should_check_block = len(value_block) >= rolling_window_size

        # If we have enough values in the block, check whether the value change
        # is at or below the threshold and apply symbolic rules if so.
        if should_check_block:
            value_delta = value_block[-1] - value_block[0]

            if value_delta <= value_skip_threshold:
                # Ask the environment for suggested rule-based actions.
                suggested_actions = envs.envs[0].guide_agent()
                valid_suggested_actions = [a for a in suggested_actions if a is not None]

                if valid_suggested_actions:
                    chosen_action = int(random.choice(valid_suggested_actions))
                else:
                    # Fallback: sample from actions not executed in this window.
                    num_actions = envs.single_action_space.n
                    excluded = set(action_block[-rolling_window_size:])
                    special_excludes = {4} # Exclude None in WaterWorld
                    candidate_actions = [a for a in range(num_actions) if a not in excluded and a not in special_excludes]
                    if not candidate_actions:
                        candidate_actions = [a for a in range(num_actions) if a not in special_excludes]

                    chosen_action = int(random.choice(candidate_actions))
                    
            # Slide the rolling window forward.
            value_block.pop(0)
            action_block.pop(0)

        actions = torch.tensor([chosen_action], dtype=torch.long, device=device)

        episode_actions.append(float(chosen_action))
        episode_logprobs.append(logprob_val)
        episode_entropies.append(entropy_val)
        episode_values.append(value_val)

        per_step_metrics.append(
            {
                "Episode": len(episodic_returns) + 1,
                "Step": current_episode_step,
                "Action": float(chosen_action),
                "LogProb": logprob_val,
                "Entropy": entropy_val,
                "Value": value_val,
            }
        )
        total_eval_steps += 1

        if track:
            writer.add_scalar("eval_step/action", float(chosen_action), total_eval_steps)
            writer.add_scalar("eval_step/logprob", logprob_val, total_eval_steps)
            writer.add_scalar("eval_step/entropy", entropy_val, total_eval_steps)
            writer.add_scalar("eval_step/value", value_val, total_eval_steps)
            writer.add_scalar("eval_step/episode", len(episodic_returns) + 1, total_eval_steps)
            writer.add_scalar("eval_step/episode_step", current_episode_step, total_eval_steps)

        next_obs, _, done, infos = envs.step(actions.cpu().numpy())

        for info in infos:
            if "episode" in info:
                episode_action_mean = float(np.mean(episode_actions))
                episode_action_std = float(np.std(episode_actions))
                episode_logprob_mean = float(np.mean(episode_logprobs))
                episode_logprob_std = float(np.std(episode_logprobs))
                episode_entropy_mean = float(np.mean(episode_entropies))
                episode_entropy_std = float(np.std(episode_entropies))
                episode_value_mean = float(np.mean(episode_values))
                episode_value_std = float(np.std(episode_values))

                episodic_returns += [info["episode"]["r"]]
                episodic_actions_mean.append(episode_action_mean)
                episodic_actions_std.append(episode_action_std)
                episodic_logprobs_mean.append(episode_logprob_mean)
                episodic_logprobs_std.append(episode_logprob_std)
                episodic_entropies_mean.append(episode_entropy_mean)
                episodic_entropies_std.append(episode_entropy_std)
                episodic_values_mean.append(episode_value_mean)
                episodic_values_std.append(episode_value_std)
                pbar.update(1)

                if len(episodic_returns) % 100 == 0:
                    tqdm.write(
                        f"eval_episode={len(episodic_returns)}, "
                        f"episodic_return={info['episode']['r']}"
                    )

                if track:
                    writer.add_scalar("eval/episodic_return", info["episode"]["r"], len(episodic_returns))
                    writer.add_scalar("eval/episodic_action_mean", episode_action_mean, len(episodic_returns))
                    writer.add_scalar("eval/episodic_action_std", episode_action_std, len(episodic_returns))
                    writer.add_scalar("eval/episodic_logprob_mean", episode_logprob_mean, len(episodic_returns))
                    writer.add_scalar("eval/episodic_logprob_std", episode_logprob_std, len(episodic_returns))
                    writer.add_scalar("eval/episodic_entropy_mean", episode_entropy_mean, len(episodic_returns))
                    writer.add_scalar("eval/episodic_entropy_std", episode_entropy_std, len(episodic_returns))
                    writer.add_scalar("eval/episodic_value_mean", episode_value_mean, len(episodic_returns))
                    writer.add_scalar("eval/episodic_value_std", episode_value_std, len(episodic_returns))

                episode_actions = []
                episode_logprobs = []
                episode_entropies = []
                episode_values = []
                value_block = []
                action_block = []

                if len(episodic_returns) < eval_episodes:
                    envs.seed(seeds[len(episodic_returns)])
                    next_obs = envs.reset()

        obs = next_obs

    if track:
        writer.close()

    if return_episode_metrics:
        return episodic_returns, {
            "per_step": per_step_metrics,
            "per_episode": {
                "action_mean": episodic_actions_mean,
                "action_std": episodic_actions_std,
                "logprob_mean": episodic_logprobs_mean,
                "logprob_std": episodic_logprobs_std,
                "entropy_mean": episodic_entropies_mean,
                "entropy_std": episodic_entropies_std,
                "value_mean": episodic_values_mean,
                "value_std": episodic_values_std,
            },
        }

    return episodic_returns


if __name__ == "__main__":
    from ppo import make_env, Agent
    args = tyro.cli(Args)
    env_id = args.env_id
    model_path = args.model_path
    eval_episodes = args.eval_episodes
    run_name = f"{args.task}__{args.exp_name}__{args.seed}__{int(time.time())}-eval"
    capture_video = args.capture_video
    seed = args.seed

    evaluate(
        model_path,
        make_env,
        env_id,
        eval_episodes=eval_episodes,
        run_name=run_name,
        seed=seed,
        group_name=args.group_name,
        Model=Agent,
        device="cpu",
        capture_video=capture_video,
        track=args.track,
    )
