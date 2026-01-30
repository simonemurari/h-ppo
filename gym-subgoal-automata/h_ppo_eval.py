import os
from typing import Callable
import tyro
import time
import gym
import wandb
import torch
import numpy as np
import random
import csv
from tqdm import tqdm
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from config_eval import Args
import warnings
warnings.filterwarnings("ignore")

def heuristic_action_selection(agent, envs, x, epsilon):
        logits = agent.actor(x)
        probs = Categorical(logits=logits)
        
        multiplier = torch.ones_like(logits)
        for i in range(x.shape[0]):
            suggested_actions_list = envs.envs[i].guide_agent()
            for suggested_action in suggested_actions_list:
                if suggested_action is not None:
                    multiplier[i, suggested_action] += agent.conf_level * epsilon
        modified_probs = probs.probs * multiplier
        modified_probs = modified_probs / modified_probs.sum(dim=-1, keepdim=True)
        probs = Categorical(probs=modified_probs)
        
        action = probs.sample()

        return action

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    seed: int,
    group_name: str,
    epsilon: float,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    track: bool = False,
    from_config: bool = True,
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
        epsilon = args.epsilon
        wandb_project_name = args.wandb_project_name
        wandb_entity = args.wandb_entity

    if epsilon is None:
        raise ValueError("Epsilon must be provided for heuristic PPO evaluation.")

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
    envs.seed(seeds[0])
    obs = envs.reset()
    episodic_returns = []
    pbar = tqdm(total=eval_episodes)
    tqdm.write(f"Model path: {model_path}")
    tqdm.write(f"Evaluating {env_id} with {eval_episodes} episodes")

    while len(episodic_returns) < eval_episodes:
        
        actions = heuristic_action_selection(agent, envs, torch.Tensor(obs).to(device), epsilon)

        next_obs, _, done, infos = envs.step(actions.cpu().numpy())
        for info in infos:
            if "episode" in info:
                episodic_returns += [info["episode"]["r"]]
                pbar.update(1)
                if len(episodic_returns) % 100 == 0:
                    tqdm.write(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                if track:
                    writer.add_scalar("eval/episodic_return", info["episode"]["r"], len(episodic_returns))
                # Re-seed and reset for next episode
                if len(episodic_returns) < eval_episodes:
                    envs.seed(seeds[len(episodic_returns)])
                    next_obs = envs.reset()
        obs = next_obs

    if track:
        writer.close()

    return episodic_returns


if __name__ == "__main__":
    # from huggingface_hub import hf_hub_download

    from h_ppo_product import make_env, Agent
    args = tyro.cli(Args)
    env_id = args.env_id
    model_path = args.model_path
    eval_episodes = args.eval_episodes
    run_name = f"{args.task}__{args.exp_name}__{args.seed}__{int(time.time())}-eval"
    capture_video = args.capture_video
    seed = args.seed
    epsilon = args.epsilon

    # model_path = hf_hub_download(
    #     repo_id="sdpkjc/Hopper-v4-ppo_continuous_action-seed1", filename="ppo_continuous_action.cleanrl_model"
    # )
    evaluate(
        model_path,
        make_env,
        env_id,
        eval_episodes=eval_episodes,
        run_name=run_name,
        seed=seed,
        group_name=args.group_name,
        epsilon=args.epsilon,
        Model=Agent,
        device="cpu",
        capture_video=capture_video,
        track=args.track,
    )
