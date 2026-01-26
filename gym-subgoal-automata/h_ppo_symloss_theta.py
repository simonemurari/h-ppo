# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import random
import time
from datetime import datetime
import sys
from tqdm import tqdm, trange
import tyro
from pathlib import Path
import os
import math
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.append(str(Path(__file__).parent.parent))
from config import Args
import gym
import gym.wrappers
from gym import spaces
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
import warnings

load_dotenv(".env")
WANDB_KEY = os.getenv("WANDB_KEY")
warnings.filterwarnings("ignore")
print(
    f"Torch: {torch.__version__}, cuda.is_available(): {torch.cuda.is_available()}"
)


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(
            env_id,
            params={
                "generation": "random",
                "environment_seed": seed,
                "use_one_hot_vector_states": True,
            },
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if isinstance(env.observation_space, spaces.Discrete):
            n = env.observation_space.n
            env.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(n,), dtype=np.float32
            )
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
        self.conf_level = 0.8
        self.num_actions = envs.single_action_space.n

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def get_logprob_symbolic(self, x, action, envs):
        torch_probs = torch.ones((x.shape[0], self.num_actions), device=x.device) * (1.0 - self.conf_level)
        for i in range(x.shape[0]):
            suggested_actions_list = envs.envs[i].guide_agent()
            for suggested_action in suggested_actions_list:
                if suggested_action is not None:
                    torch_probs[i, suggested_action] = self.conf_level
        torch_probs = torch_probs / torch_probs.sum(dim=-1, keepdim=True)
        probs = Categorical(probs=torch_probs)
        return probs.log_prob(action)


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Compute runtime values
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    run_name = f"{args.task}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=f"h_ppo_symloss_theta_{args.group_name}"
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = not args.torch_deterministic
        print(f"Using {device} device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using {device} device")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_symbolic = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    envs.seed(args.seed)
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    episodes_returns = []
    episodes_lengths = []
    len_ep_ret = 0

    for iteration in trange(1, args.num_iterations + 1, colour="green"):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                logprob_symbolic = agent.get_logprob_symbolic(next_obs, action, envs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            logprobs_symbolic[step] = logprob_symbolic

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    writer.add_scalar("episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("episodic_length", item["episode"]["l"], global_step)
                    episodes_returns.append(item["episode"]["r"])
                    episodes_lengths.append(item["episode"]["l"])
                    if iteration % max(1, math.ceil(args.num_iterations / 10)) == 0 or iteration == args.num_iterations:
                        old_len_ep_ret = len_ep_ret
                        len_ep_ret = len(episodes_returns)
                        num_eps = len_ep_ret - old_len_ep_ret
                        mean_episodic_return = np.mean(episodes_returns[-num_eps:])
                        mean_episodic_length = np.mean(episodes_lengths[-num_eps:])
                        tot_mean_ret = np.mean(episodes_returns)
                        tot_mean_len = np.mean(episodes_lengths)
                        tqdm.write(f"global_step={global_step}, mean_episodic_return={mean_episodic_return}, mean_episodic_length={mean_episodic_length}, total_mean_return={tot_mean_ret}, total_mean_length={tot_mean_len}")

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_logprobs_symbolic = logprobs_symbolic.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                logratio_symbolic = newlogprob - b_logprobs_symbolic[mb_inds]
                ratio = logratio.exp()
                ratio_symbolic = logratio_symbolic.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    old_approx_kl_symbolic = (-logratio_symbolic).mean()
                    approx_kl_symbolic = ((ratio_symbolic - 1) - logratio_symbolic).mean()
                    clipfracs += [((ratio_symbolic - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                pg_loss1_symbolic = -mb_advantages * ratio_symbolic
                pg_loss2_symbolic = -mb_advantages * torch.clamp(ratio_symbolic, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_symbolic = torch.max(pg_loss1_symbolic, pg_loss2_symbolic).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + pg_loss_symbolic * args.theta

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        os.makedirs(f"models/h_ppo_symloss_theta_{args.task}{args.run_code}", exist_ok=True)
        model_path = f"models/h_ppo_symloss_theta_{args.task}{args.run_code}/h_ppo_symloss_theta_seed={args.seed}.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
