# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import math
import os
import random
import sys
import time
import minigrid
from config import Args
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from doorkey_helpers import apply_rules_batch, get_grounded_predicate_vectors, get_observables
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange


def make_env(env_id, n_keys, idx, capture_video, run_name, random_color=True):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, random_color=random_color, render_mode="rgb_array", n_keys=n_keys)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, n_keys=n_keys, random_color=random_color)
            env = gym.wrappers.FlattenObservation(
                gym.wrappers.FilterObservation(env, filter_keys=["image", "direction"])
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def total_variation_distance_matrix(p1, p2):
    """Pairwise TV distance matrix between two sets of action distributions.
    p1: (N, A)
    p2: (M, A)
    returns: (N, M)
    """
    return 0.5 * torch.abs(p1[:, None, :] - p2[None, :, :]).sum(dim=-1)


def build_transition_terminals(dones):
    """Marks whether the transition out of each rollout step is terminal.

    In CleanRL-style storage, `dones[t]` is the done flag *before* stepping from
    `obs[t]`, so transition-terminal flags must be shifted by one step.
    """
    transition_terminals = torch.zeros_like(dones, dtype=torch.bool)
    if dones.shape[0] > 1:
        transition_terminals[:-1] = dones[1:].bool()
    transition_terminals[-1] = True
    return transition_terminals


def compute_policy_similarity_metric_fixed_point(
    policy_distance_matrix,
    has_successor_x,
    has_successor_y,
    gamma,
    dp_eps,
):
    """Fixed-point dynamic-programming solver for deterministic D.2 recursion.

    This follows the same style used in jumping_task (`while True` fixed-point
    iteration), while preserving terminal-transition masking for rollout data.
    """
    num_x, num_y = policy_distance_matrix.shape
    if num_x == 0 or num_y == 0:
        return policy_distance_matrix

    # Paper D.2 convention: DIST between terminal states is zero.
    terminal_pair_mask = (~has_successor_x)[:, None] & (~has_successor_y)[None, :]
    effective_policy_distance = torch.where(
        terminal_pair_mask,
        torch.zeros_like(policy_distance_matrix),
        policy_distance_matrix,
    )

    current_metric = torch.zeros_like(policy_distance_matrix)
    next_x = (torch.arange(num_x, device=policy_distance_matrix.device) + 1).clamp(max=num_x - 1)
    next_y = (torch.arange(num_y, device=policy_distance_matrix.device) + 1).clamp(max=num_y - 1)
    future_mask = has_successor_x[:, None] & has_successor_y[None, :]
    zero_future = torch.zeros_like(policy_distance_matrix)

    fixed_point_eps = float(dp_eps)

    while True:
        shifted_future = current_metric[next_x][:, next_y]
        updated_metric = effective_policy_distance + gamma * torch.where(
            future_mask,
            shifted_future,
            zero_future,
        )
        if torch.sum(torch.abs(updated_metric - current_metric)).item() < fixed_point_eps:
            return updated_metric
        current_metric = updated_metric


def compute_psm_contrastive_loss(similarity_matrix, gamma_matrix, temperature):
    """Bidirectional PSM contrastive loss (paper Eq. 4, both directions).

    Ported from h_ppo_PSM_v1 but keeps grounded predicate vectors as similarity
    (no gradient flows into the actor).

    similarity_matrix: (N, M) cosine similarity of grounded vectors
    gamma_matrix:      (N, M) soft coupling Γ(x, y) = exp(-d*/β)
    temperature:       inverse temperature λ
    """
    N, M = similarity_matrix.shape
    if N == 0 or M == 0:
        return similarity_matrix.new_zeros(())

    scaled_sim = similarity_matrix / max(temperature, 1e-8)

    # Direction 1: for each y (column), positive x = argmax_x Γ(x, y)
    pos_x_idx = torch.argmax(gamma_matrix, dim=0)              # (M,)
    col_range = torch.arange(M, device=similarity_matrix.device)
    pos_logits_1 = scaled_sim[pos_x_idx, col_range]
    pos_weight_1 = gamma_matrix[pos_x_idx, col_range].clamp_min(1e-8)
    logits_1 = scaled_sim + torch.log((1.0 - gamma_matrix).clamp_min(1e-8))
    logits_1 = logits_1.clone()
    logits_1[pos_x_idx, col_range] = pos_logits_1 + torch.log(pos_weight_1)
    loss_1 = (torch.logsumexp(logits_1, dim=0) - (pos_logits_1 + torch.log(pos_weight_1))).mean()

    # Direction 2: for each x (row), positive y = argmax_y Γ(x, y)
    pos_y_idx = torch.argmax(gamma_matrix, dim=1)              # (N,)
    row_range = torch.arange(N, device=similarity_matrix.device)
    pos_logits_2 = scaled_sim[row_range, pos_y_idx]
    pos_weight_2 = gamma_matrix[row_range, pos_y_idx].clamp_min(1e-8)
    logits_2 = scaled_sim + torch.log((1.0 - gamma_matrix).clamp_min(1e-8))
    logits_2 = logits_2.clone()
    logits_2[row_range, pos_y_idx] = pos_logits_2 + torch.log(pos_weight_2)
    loss_2 = (torch.logsumexp(logits_2, dim=1) - (pos_logits_2 + torch.log(pos_weight_2))).mean()

    return loss_1 + loss_2


def precompute_symbolic_to_neural_psm_rollout_cache(
    agent,
    b_obs,
    rollout_symbolic_probs,
    rollout_transition_terminals,
    num_steps,
    num_envs,
    gamma,
    dp_eps,
    beta,
):
    """Pre-computes symbolic->neural D.2 metrics for all ordered env pairs once per rollout."""
    device = b_obs.device
    rollout_neural_probs = agent.get_policy_probs(b_obs).reshape(num_steps, num_envs, -1)
    has_successor = torch.arange(num_steps, device=device) < (num_steps - 1)
    transition_terminals = rollout_transition_terminals.bool()

    psm_similarity_by_pair = {}
    metric_mean_by_pair = {}
    metric_means = []

    for source_env in range(num_envs):
        symbolic_x = rollout_symbolic_probs[:, source_env, :]
        active_x = has_successor & (~transition_terminals[:, source_env])

        for target_env in range(num_envs):
            if target_env == source_env:
                continue

            neural_y = rollout_neural_probs[:, target_env, :]
            active_y = has_successor & (~transition_terminals[:, target_env])

            dist_matrix = total_variation_distance_matrix(symbolic_x, neural_y)
            metric_matrix = compute_policy_similarity_metric_fixed_point(
                policy_distance_matrix=dist_matrix,
                has_successor_x=active_x,
                has_successor_y=active_y,
                gamma=gamma,
                dp_eps=dp_eps,
            )
            pair_key = (source_env, target_env)
            psm_similarity_by_pair[pair_key] = torch.exp(-metric_matrix / max(beta, 1e-8))
            metric_mean_by_pair[pair_key] = metric_matrix.mean()
            metric_means.append(metric_mean_by_pair[pair_key])

    if metric_means:
        rollout_metric_mean = torch.stack(metric_means).mean()
    else:
        rollout_metric_mean = b_obs.new_zeros(())

    return psm_similarity_by_pair, metric_mean_by_pair, rollout_metric_mean


def compute_symbolic_to_neural_psm_auxiliary_loss(
    agent,
    b_obs,
    psm_similarity_by_pair,
    metric_mean_by_pair,
    mb_inds,
    num_envs,
    num_steps,
    temperature,
    pair_cycle_index,
):
    """Computes PSM contrastive loss for one minibatch using the full rollout matrices.

    mb_inds is used only for deterministic pair cycling; the loss itself operates
    on the full (num_steps × num_steps) precomputed matrices, not a subset.
    """
    if num_envs < 2 or not psm_similarity_by_pair:
        zero = b_obs.new_zeros(())
        return zero, zero, pair_cycle_index

    # Determine which env pair to use via deterministic cycling
    mb_envs = torch.remainder(torch.as_tensor(mb_inds, dtype=torch.long), num_envs)
    target_env_candidates = torch.unique(mb_envs).tolist()
    if not target_env_candidates:
        zero = b_obs.new_zeros(())
        return zero, zero, pair_cycle_index

    target_slot = pair_cycle_index % len(target_env_candidates)
    sampled_target_env = target_env_candidates[target_slot]

    source_env_candidates = [e for e in range(num_envs) if e != sampled_target_env]
    if not source_env_candidates:
        zero = b_obs.new_zeros(())
        return zero, zero, pair_cycle_index
    source_slot = (pair_cycle_index // max(1, len(target_env_candidates))) % len(source_env_candidates)
    sampled_source_env = source_env_candidates[source_slot]

    pair_key = (sampled_source_env, sampled_target_env)
    if pair_key not in psm_similarity_by_pair:
        zero = b_obs.new_zeros(())
        return zero, zero, pair_cycle_index + 1

    # Compute the learnable similarity s_theta from grounded vectors passed
    # through a small auxiliary projection head.
    src_obs_indices = torch.arange(num_steps, device=b_obs.device) * num_envs + sampled_source_env
    tgt_obs_indices = torch.arange(num_steps, device=b_obs.device) * num_envs + sampled_target_env

    src_embedding = agent.get_grounded_embedding(b_obs[src_obs_indices])
    tgt_embedding = agent.get_grounded_embedding(b_obs[tgt_obs_indices])
    src_embedding = src_embedding / (src_embedding.norm(dim=-1, keepdim=True) + 1e-8)
    tgt_embedding = tgt_embedding / (tgt_embedding.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_matrix = src_embedding @ tgt_embedding.T

    # Bidirectional loss on the full (num_steps × num_steps) precomputed matrices.
    pair_loss = compute_psm_contrastive_loss(
        similarity_matrix=similarity_matrix,
        gamma_matrix=psm_similarity_by_pair[pair_key],
        temperature=temperature,
    )

    return pair_loss, metric_mean_by_pair[pair_key], pair_cycle_index + 1


class Agent(nn.Module):
    """PPO actor-critic with symbolic policy helpers for PSM."""
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
        self.grounded_projection = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64), std=1.0),
        )
        self.conf_level = 0.8
        self.num_actions = envs.single_action_space.n

    def get_value(self, x):
        """Returns V(s)."""
        return self.critic(x)

    def get_policy_probs(self, x):
        """Returns learned action probabilities pi(a|s)."""
        return torch.softmax(self.actor(x), dim=-1)

    def get_grounded_vectors(self, x):
        """Returns deterministic grounded predicate vectors from observables."""
        grounded_vectors = get_grounded_predicate_vectors(x[:, 4:])
        return torch.as_tensor(grounded_vectors, device=x.device, dtype=x.dtype)

    def get_grounded_embedding(self, x):
        """Returns a learnable projection of grounded vectors for the PSM loss."""
        return self.grounded_projection(self.get_grounded_vectors(x))

    def get_symbolic_probs(self, x):
        """Builds symbolic action distributions from rule-based observables.

        This mirrors the symbolic rule machinery used in `h_ppo_symloss.py`.
        """
        suggested_actions_batch = apply_rules_batch(get_observables(x[:, 4:]))
        symbolic_probs = torch.ones((x.shape[0], self.num_actions), device=x.device, dtype=x.dtype)
        symbolic_probs *= 1.0 - self.conf_level
        for batch_idx, suggested_actions in enumerate(suggested_actions_batch):
            for suggested_action in suggested_actions:
                if suggested_action is not None:
                    symbolic_probs[batch_idx, suggested_action] = self.conf_level
        return symbolic_probs / symbolic_probs.sum(dim=-1, keepdim=True)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
            group=f"h_ppo_PSM_{args.group_name}"
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
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = not args.torch_deterministic
        print(f"Using {device} device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using {device} device")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.n_keys, i, args.capture_video, run_name, args.random_color) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    episodes_returns = []
    episodes_lengths = []
    len_ep_ret = 0

    for iteration in trange(1, args.num_iterations + 1, colour="green"):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("episodic_length", info["episode"]["l"], global_step)
                        episodes_returns.append(info["episode"]["r"])
                        episodes_lengths.append(info["episode"]["l"])
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
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        rollout_symbolic_probs = None
        rollout_psm_similarity_by_pair = None
        rollout_metric_mean_by_pair = None
        rollout_transition_terminals = build_transition_terminals(dones)

        psm_cme_running = torch.tensor(0.0, device=device)
        psm_metric_running = torch.tensor(0.0, device=device)
        psm_cme_count = 0
        psm_pair_cycle_index = 0

        if args.psm_aux_coef > 0:
            with torch.no_grad():
                # Cached once per rollout to avoid re-running symbolic/grounded parsing
                # inside every minibatch step.
                rollout_symbolic_probs = agent.get_symbolic_probs(b_obs).reshape(args.num_steps, args.num_envs, -1)
                (
                    rollout_psm_similarity_by_pair,
                    rollout_metric_mean_by_pair,
                    _,
                ) = precompute_symbolic_to_neural_psm_rollout_cache(
                    agent=agent,
                    b_obs=b_obs,
                    rollout_symbolic_probs=rollout_symbolic_probs,
                    rollout_transition_terminals=rollout_transition_terminals,
                    num_steps=args.num_steps,
                    num_envs=args.num_envs,
                    gamma=args.gamma,
                    dp_eps=args.psm_dp_eps,
                    beta=args.psm_beta,
                )

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
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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
                mb_psm_cme_loss = torch.tensor(0.0, device=device)
                if (
                    args.psm_aux_coef > 0
                    and rollout_psm_similarity_by_pair is not None
                    and rollout_metric_mean_by_pair is not None
                ):
                    mb_psm_cme_loss, mb_psm_metric_mean, psm_pair_cycle_index = compute_symbolic_to_neural_psm_auxiliary_loss(
                        agent,
                        b_obs,
                        rollout_psm_similarity_by_pair,
                        rollout_metric_mean_by_pair,
                        mb_inds,
                        num_envs=args.num_envs,
                        num_steps=args.num_steps,
                        temperature=args.psm_temperature,
                        pair_cycle_index=psm_pair_cycle_index,
                    )
                    psm_cme_running = psm_cme_running + mb_psm_cme_loss.detach()
                    psm_metric_running = psm_metric_running + mb_psm_metric_mean.detach()
                    psm_cme_count += 1
                # Total loss = L_RL + α · L_CME  (paper Section I.3, official train_step)
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + args.psm_aux_coef * mb_psm_cme_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        psm_loss = psm_cme_running / max(psm_cme_count, 1)
        psm_metric_mean = psm_metric_running / max(psm_cme_count, 1)

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
        writer.add_scalar("losses/psm_cme", psm_loss.item(), global_step)
        writer.add_scalar("losses/psm_metric_mean", psm_metric_mean.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

    if args.save_model:
        os.makedirs(f"models/h_ppo_PSM_{args.size_env}x{args.size_env}_{args.n_keys}keys{args.run_code}", exist_ok=True)
        model_path = f"models/h_ppo_PSM_{args.size_env}x{args.size_env}_{args.n_keys}keys{args.run_code}/h_ppo_PSM_seed={args.seed}.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        # Prevent tyro in the evaluation module from parsing the training CLI args
        saved_argv = sys.argv.copy()
        try:
            sys.argv = [sys.argv[0]]
            from ppo_eval import evaluate

            episodic_returns = evaluate(
                model_path,
                make_env,
                args.env_id,
                args.n_keys,
                eval_episodes=1000,
                run_name=f"{run_name}-eval",
                seed=args.seed,
                group_name=f"h_ppo_PSM_{args.group_name}_evals",
                Model=Agent,
                device=device,
                track=False,
                from_config=False,
                random_color=args.random_color
            )
        finally:
            sys.argv = saved_argv
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # if args.upload_model:
        #     from cleanrl_utils.huggingface import push_to_hub

        #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
        #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #     push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")


    envs.close()
    writer.close()