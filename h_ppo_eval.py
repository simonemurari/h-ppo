import os
from typing import Callable
import tyro
import time
import gymnasium as gym
import minigrid
import wandb
import torch
import numpy as np
import random
import csv
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
from config_eval import Args
from doorkey_helpers import get_observables, apply_rules_batch, action_map

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    n_keys: int,
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
    random_color: bool = True
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
        epsilon = args.epsilon

    if epsilon is None:
        raise ValueError("Epsilon must be provided for heuristic PPO evaluation.")

    if track:

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
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

    obs, _ = envs.reset(seed=seed)
    episodic_returns = []
    pbar = trange(eval_episodes)
    tqdm.write(f"Model path: {model_path}")
    tqdm.write(f"Evaluating {env_id} with {n_keys} keys, seed={seed}")
    
    sum_actions = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    match_actions = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    step_actions = {}
    i = 0

    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device), apply_heuristic=True, epsilon=epsilon)
        
        suggested_actions = apply_rules_batch(get_observables(obs[:, 4:]))
        if actions[0] in suggested_actions[0]:
            sum_actions[int(actions[0])] += 1
            match_actions[int(actions[0])] += 1
            match = 1
        elif suggested_actions[0][0] is not None:
            sum_actions[int(actions[0])] += 1
            match = 0
        else:
            match = -1  # no suggested action
        step_actions[i] = (int(actions[0]), suggested_actions[0], match)
        if len(suggested_actions) > 1:
            print(f"step_actions: {step_actions[i]}")
        if suggested_actions[0][0] is not None:
            actions = np.random.choice(suggested_actions[0], size=1)
            actions = torch.Tensor(actions).long().to(device)
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                episodic_returns += [info["episode"]["r"]]
                if len(episodic_returns) % 100 == 0:
                    tqdm.write(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                if track:
                    writer.add_scalar("eval/episodic_return", info["episode"]["r"], len(episodic_returns))
        obs = next_obs
        pbar.update(1)
        i += 1

    if track:
        writer.close()

    print(f"\n\n---Action match statistics for seed {seed}---:")
    total_taken = sum(sum_actions.values())
    total_matched = sum(match_actions.values())
    overall_match_rate = (total_matched / total_taken * 100) if total_taken > 0 else 0.0
    print(f"Overall match rate: {overall_match_rate:.2f}% ({total_matched}/{total_taken})")
    for action in sum_actions.keys():
        total_action = sum_actions[action]
        match_action = match_actions[action]
        match_rate = (match_action / total_action * 100) if total_action > 0 else 0.0
        action_name = [k for k, v in action_map.items() if v == action]
        action_name = action_name[0] if action_name else "unused action"
        print(f"Action {action} ({action_name}): {match_action}/{total_action} matches ({match_rate:.2f}%)")

    # # Write to CSV
    # general_folder_name = 'eval_results_hppo'
    # os.makedirs(f'{general_folder_name}/{group_name}', exist_ok=True)
    # csv_filename = f"{general_folder_name}/{group_name}/seed={seed}_omr={overall_match_rate:.2f}.csv"
    # with open(csv_filename, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['action', 'network_total', 'rules_total', 'match_rate'])
    #     for action in [0, 1, 2, 3, 5]:
    #         total_action = sum_actions[action]
    #         match_action = match_actions[action]
    #         match_rate = (match_action / total_action * 100) if total_action > 0 else 0.0
    #         action_name = [k for k, v in action_map.items() if v == action][0]
    #         writer.writerow([action_name, total_action, match_action, f"{match_rate:.2f}"])

    # # Write to CSV
    # csv_filename = f"{general_folder_name}/{group_name}/seed={seed}_STEPACTIONS.csv"
    # with open(csv_filename, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['step', 'network_action', 'rules_suggested_actions', 'match'])
    #     for episode, (net_action, rule_actions, match) in step_actions.items():
    #         rule_actions_str = ';'.join([str(a) for a in rule_actions])
    #         writer.writerow([episode, net_action, rule_actions_str, match])

    return episodic_returns


if __name__ == "__main__":
    # from huggingface_hub import hf_hub_download

    from h_ppo_product import make_env, Agent
    args = tyro.cli(Args)
    env_id = args.env_id
    n_keys = args.n_keys
    model_path = args.model_path
    eval_episodes = args.eval_episodes
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}-eval"
    capture_video = args.capture_video
    seed = args.seed
    random_color = args.random_color
    epsilon = args.epsilon

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
        epsilon=args.epsilon,
        Model=Agent,
        device="cpu",
        capture_video=capture_video,
        track=args.track,
        random_color=random_color
    )