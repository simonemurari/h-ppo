from typing import Callable
import tyro
import time
import gymnasium as gym
import wandb
import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from config_eval import Args

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    n_keys: int,
    eval_episodes: int,
    run_name: str,
    seed: int,
    group_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    track: bool = False,
    from_config: bool = True,
    random_color: bool = True,
    return_episode_metrics: bool = False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    envs = gym.vector.SyncVectorEnv([make_env(env_id, n_keys, 0, capture_video, run_name, random_color)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    if from_config is True:
        args = tyro.cli(Args)
        env_id = args.env_id
        model_path = args.model_path
        eval_episodes = args.eval_episodes
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}-eval"
        group_name = args.group_name
        capture_video = args.capture_video
        seed = args.seed
        track = args.track
        n_keys = args.n_keys
        wandb_project_name = args.wandb_project_name
        wandb_entity = args.wandb_entity

    if track:

        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            sync_tensorboard=True,
            config=vars(args) if from_config else {},
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=group_name
        )

        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    seeds = random.sample(range(0, 10**9), eval_episodes)
    obs, _ = envs.reset(seed=seeds[0])
    episodic_returns = []
    episodic_actions = []
    episodic_logprobs = []
    episodic_entropies = []
    episodic_values = []
    current_action = 0.0
    current_logprob = 0.0
    current_entropy = 0.0
    current_value = 0.0
    pbar = tqdm(total=eval_episodes)
    tqdm.write(f"Model path: {model_path}")
    tqdm.write(f"Evaluating {env_id} with {n_keys} keys, {eval_episodes} episodes")

    while len(episodic_returns) < eval_episodes:
        actions, logprobs, entropies, values = agent.get_action_and_value(torch.Tensor(obs).to(device))
        current_action = float(actions.item())
        current_logprob = float(logprobs.item())
        current_entropy = float(entropies.item())
        current_value = float(values.item())

        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                episodic_returns += [info["episode"]["r"]]
                episodic_actions.append(current_action)
                episodic_logprobs.append(current_logprob)
                episodic_entropies.append(current_entropy)
                episodic_values.append(current_value)
                pbar.update(1)
                if len(episodic_returns) % 100 == 0:
                    tqdm.write(f"eval_episode={len(episodic_returns)}, mean_episodic_return={np.mean(episodic_returns):.2f}")
                if track:
                    writer.add_scalar("eval/episodic_return", info["episode"]["r"], len(episodic_returns))
                    writer.add_scalar("eval/episodic_action", episodic_actions[-1], len(episodic_returns))
                    writer.add_scalar("eval/episodic_logprob", episodic_logprobs[-1], len(episodic_returns))
                    writer.add_scalar("eval/episodic_entropy", episodic_entropies[-1], len(episodic_returns))
                    writer.add_scalar("eval/episodic_value", episodic_values[-1], len(episodic_returns))
                
                if len(episodic_returns) < eval_episodes:
                    next_obs, _ = envs.reset(seed=seeds[len(episodic_returns)])
        obs = next_obs

    if track:
        writer.close()

    if return_episode_metrics:
        return episodic_returns, {
            "action": episodic_actions,
            "logprob": episodic_logprobs,
            "entropy": episodic_entropies,
            "value": episodic_values,
        }

    return episodic_returns


if __name__ == "__main__":
    # from huggingface_hub import hf_hub_download

    from ppo import make_env, Agent
    args = tyro.cli(Args)
    env_id = args.env_id
    n_keys = args.n_keys
    model_path = args.model_path
    eval_episodes = args.eval_episodes
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}-eval"
    capture_video = args.capture_video
    seed = args.seed
    random_color = args.random_color

    # model_path = hf_hub_download(
    #     repo_id="sdpkjc/Hopper-v4-ppo_continuous_action-seed1", filename="ppo_continuous_action.cleanrl_model"
    # )
    evaluate(
        model_path,
        make_env,
        env_id,
        n_keys=n_keys,
        eval_episodes=eval_episodes,
        run_name=run_name,
        seed=seed,
        group_name=args.group_name,
        Model=Agent,
        device="cpu",
        capture_video=capture_video,
        track=args.track,
        random_color=random_color
    )