# h_ppo_PSM_netembed.py — Corrected PSM implementation
#
# Key changes vs h_ppo_PSM.py:
#   1. s_θ(x,y) uses the actor's penultimate hidden layer (learnable, 64-dim)
#      instead of hand-crafted grounded vectors (no gradient path)
#   2. The contrastive loss is BIDIRECTIONAL (symmetric), matching the paper's
#      reference code  (loss1 + loss2)
#   3. Embedding computation happens WITH gradients so the PSM loss shapes
#      the actor's representation
#   4. PSM metric (symbolic-vs-neural TV distance) stays under torch.no_grad()
#      since it only defines the coupling weights Γ, not the similarity s_θ
#
import math
import os
import random
import sys
import time

from config import Args
import gymnasium as gym
import minigrid
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from doorkey_helpers import apply_rules_batch, get_observables
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange


# ──────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# PSM metric helpers  (no learnable params)
# ──────────────────────────────────────────────

def total_variation_distance_matrix(p1, p2):
    """Pairwise TV distance between two sets of action distributions.
    p1: (N, A)  p2: (M, A)  →  (N, M)
    """
    return 0.5 * torch.abs(p1[:, None, :] - p2[None, :, :]).sum(dim=-1)


def build_transition_terminals(dones):
    """Marks whether the transition out of each rollout step is terminal.

    In CleanRL storage, dones[t] is the done flag *before* the step from obs[t],
    so we shift by one to get the "transition-terminal" flag.
    """
    transition_terminals = torch.zeros_like(dones, dtype=torch.bool)
    if dones.shape[0] > 1:
        transition_terminals[:-1] = dones[1:].bool()
    transition_terminals[-1] = True
    return transition_terminals


def compute_psm_metric_fixed_point(policy_distance_matrix,
                                   has_successor_x,
                                   has_successor_y,
                                   gamma, dp_eps):
    """Fixed-point DP for deterministic PSM recursion (paper Eq. D.2).

    d*(x, y) = DIST(π_X(x), π_Y(y)) + γ · d*(x', y')

    Terminal states get DIST = 0 (paper convention).
    """
    num_x, num_y = policy_distance_matrix.shape
    if num_x == 0 or num_y == 0:
        return policy_distance_matrix

    # Zero out distance for terminal-terminal pairs (paper D.2)
    terminal_pair = (~has_successor_x)[:, None] & (~has_successor_y)[None, :]
    dist = torch.where(terminal_pair,
                       torch.zeros_like(policy_distance_matrix),
                       policy_distance_matrix)

    # Next-state indices (clamped at boundary = absorbing terminal convention)
    device = policy_distance_matrix.device
    next_x = (torch.arange(num_x, device=device) + 1).clamp(max=num_x - 1)
    next_y = (torch.arange(num_y, device=device) + 1).clamp(max=num_y - 1)
    future_mask = has_successor_x[:, None] & has_successor_y[None, :]

    current = torch.zeros_like(policy_distance_matrix)
    while True:
        future = current[next_x][:, next_y]
        updated = dist + gamma * torch.where(future_mask, future,
                                             torch.zeros_like(future))
        if torch.sum(torch.abs(updated - current)).item() < dp_eps:
            return updated
        current = updated


# ──────────────────────────────────────────────
# Contrastive loss  (learnable — gradients flow through s_θ)
# ──────────────────────────────────────────────

def compute_psm_contrastive_loss(similarity_matrix, gamma_matrix, temperature):
    """Bidirectional PSM contrastive loss (paper Eq. 4, both directions).

    Args:
        similarity_matrix: (N, M) cosine similarity s_θ(x, y) — WITH gradient
        gamma_matrix:      (N, M) soft coupling Γ(x, y) = exp(-d*/β) — no gradient
        temperature:       inverse temperature λ

    Returns scalar loss = loss_x→y + loss_y→x
    """
    N, M = similarity_matrix.shape
    if N == 0 or M == 0:
        return similarity_matrix.new_zeros(())

    scaled_sim = similarity_matrix / max(temperature, 1e-8)       # (N, M)

    # ── Direction 1: for each y (column), find nearest x ──
    # Positive: x_tilde = argmax_x Γ(x, y)
    pos_x_idx = torch.argmax(gamma_matrix, dim=0)                 # (M,)
    col_range = torch.arange(M, device=similarity_matrix.device)

    pos_logits_1 = scaled_sim[pos_x_idx, col_range]               # (M,)
    pos_weight_1 = gamma_matrix[pos_x_idx, col_range].clamp_min(1e-8)

    # Negative weights for all x' ≠ x_tilde
    neg_weights_1 = torch.log((1.0 - gamma_matrix).clamp_min(1e-8))  # (N, M)
    # Build the full logits: positive gets Γ weight, negatives get (1-Γ) weight
    logits_1 = scaled_sim + neg_weights_1                          # (N, M)
    # Override the positive entry with its own weight
    logits_1 = logits_1.clone()
    logits_1[pos_x_idx, col_range] = pos_logits_1 + torch.log(pos_weight_1)

    # InfoNCE: -log(pos / sum_all)  =  logsumexp(all) - pos
    loss_1 = (torch.logsumexp(logits_1, dim=0) -
              (pos_logits_1 + torch.log(pos_weight_1))).mean()

    # ── Direction 2: for each x (row), find nearest y ──
    pos_y_idx = torch.argmax(gamma_matrix, dim=1)                  # (N,)
    row_range = torch.arange(N, device=similarity_matrix.device)

    pos_logits_2 = scaled_sim[row_range, pos_y_idx]                # (N,)
    pos_weight_2 = gamma_matrix[row_range, pos_y_idx].clamp_min(1e-8)

    neg_weights_2 = torch.log((1.0 - gamma_matrix).clamp_min(1e-8))
    logits_2 = scaled_sim + neg_weights_2
    logits_2 = logits_2.clone()
    logits_2[row_range, pos_y_idx] = pos_logits_2 + torch.log(pos_weight_2)

    loss_2 = (torch.logsumexp(logits_2, dim=1) -
              (pos_logits_2 + torch.log(pos_weight_2))).mean()

    return loss_1 + loss_2


# ──────────────────────────────────────────────
# Agent — actor split into trunk + head so we
#         can extract the 64-dim embedding z_θ
# ──────────────────────────────────────────────

class Agent(nn.Module):
    """PPO actor-critic with learned PSM embeddings from actor hidden layer."""

    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())

        # Critic  (unchanged)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor — split into trunk (shared embedding) and head (policy logits)
        # trunk:  obs → 64-dim embedding  (this IS z_θ in PSM)
        # head:   64-dim → n_actions
        self.actor_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor_head = layer_init(
            nn.Linear(64, envs.single_action_space.n), std=0.01
        )

        self.conf_level = 0.8
        self.num_actions = envs.single_action_space.n

    # ── Core RL methods ──

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor_head(self.actor_trunk(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    # ── PSM-specific methods ──

    def get_embedding(self, x):
        """Returns the actor's penultimate 64-dim representation z_θ(x).

        This is the LEARNED embedding used for s_θ in the contrastive loss.
        Gradients flow through this into the actor trunk.
        """
        return self.actor_trunk(x)

    def get_policy_probs(self, x):
        """Returns neural policy π(a|s) as a probability vector."""
        return torch.softmax(self.actor_head(self.actor_trunk(x)), dim=-1)

    def get_symbolic_probs(self, x):
        """Returns symbolic policy probabilities from doorkey rules.

        Used as the "source" policy in PSM metric computation.
        """
        suggested_actions_batch = apply_rules_batch(get_observables(x[:, 4:]))
        symbolic_probs = torch.ones(
            (x.shape[0], self.num_actions), device=x.device, dtype=x.dtype
        )
        symbolic_probs *= 1.0 - self.conf_level
        for i, suggested_actions in enumerate(suggested_actions_batch):
            for action in suggested_actions:
                if action is not None:
                    symbolic_probs[i, action] = self.conf_level
        return symbolic_probs / symbolic_probs.sum(dim=-1, keepdim=True)


# ──────────────────────────────────────────────
# Rollout-level PSM cache
# ──────────────────────────────────────────────

def precompute_psm_metrics(agent, b_obs, num_steps, num_envs,
                           transition_terminals, gamma, dp_eps, beta):
    """Precomputes the PSM metric d* and soft similarity Γ for all env pairs.

    d* uses TV distance between symbolic_π (source) and neural_π (target).
    Γ(x,y) = exp(-d*(x,y) / β).

    All computed under torch.no_grad() — these are fixed coupling weights,
    NOT the learnable similarity s_θ.
    """
    device = b_obs.device

    with torch.no_grad():
        # Compute symbolic and neural policy probs for the entire rollout
        symbolic_probs = agent.get_symbolic_probs(b_obs).reshape(
            num_steps, num_envs, -1
        )
        neural_probs = agent.get_policy_probs(b_obs).reshape(
            num_steps, num_envs, -1
        )

    has_successor = torch.arange(num_steps, device=device) < (num_steps - 1)
    term_flags = transition_terminals.bool()

    # Store Γ matrices for each (source, target) env pair
    gamma_matrices = {}   # (src, tgt) → (num_steps, num_steps) Γ values
    metric_means = {}

    for src_env in range(num_envs):
        sym_x = symbolic_probs[:, src_env, :]
        active_x = has_successor & (~term_flags[:, src_env])

        for tgt_env in range(num_envs):
            if tgt_env == src_env:
                continue

            neur_y = neural_probs[:, tgt_env, :]
            active_y = has_successor & (~term_flags[:, tgt_env])

            # TV distance: symbolic(env_X) vs neural(env_Y)
            tv_matrix = total_variation_distance_matrix(sym_x, neur_y)

            # Fixed-point PSM (paper Eq D.2)
            d_star = compute_psm_metric_fixed_point(
                tv_matrix, active_x, active_y, gamma, dp_eps
            )

            # Soft similarity Γ = exp(-d* / β)
            pair = (src_env, tgt_env)
            gamma_matrices[pair] = torch.exp(-d_star / max(beta, 1e-8))
            metric_means[pair] = d_star.mean()

    return gamma_matrices, metric_means


# ──────────────────────────────────────────────
# Minibatch PSM loss (called inside PPO update)
# ──────────────────────────────────────────────

def compute_psm_loss_for_minibatch(agent, b_obs, gamma_matrices, metric_means,
                                   mb_inds, num_envs, num_steps,
                                   temperature, pair_cycle_idx):
    """Computes PSM contrastive loss for one PPO minibatch.

    Embeddings z_θ are computed WITH gradients so the loss shapes the actor trunk.
    Coupling weights Γ are precomputed and fixed (no gradient).
    """
    device = b_obs.device
    mb_inds_t = torch.as_tensor(mb_inds, device=device, dtype=torch.long)
    mb_envs = torch.remainder(mb_inds_t, num_envs)
    mb_times = torch.div(mb_inds_t, num_envs, rounding_mode="floor")

    if num_envs < 2:
        return b_obs.new_zeros(()), b_obs.new_zeros(()), pair_cycle_idx

    # Pick one (source, target) pair per minibatch via deterministic cycling
    tgt_candidates = torch.unique(mb_envs).tolist()
    if not tgt_candidates:
        return b_obs.new_zeros(()), b_obs.new_zeros(()), pair_cycle_idx

    tgt_env = tgt_candidates[pair_cycle_idx % len(tgt_candidates)]
    src_candidates = [e for e in range(num_envs) if e != tgt_env]
    if not src_candidates:
        return b_obs.new_zeros(()), b_obs.new_zeros(()), pair_cycle_idx

    src_env = src_candidates[
        (pair_cycle_idx // max(1, len(tgt_candidates))) % len(src_candidates)
    ]

    pair = (src_env, tgt_env)
    if pair not in gamma_matrices:
        return b_obs.new_zeros(()), b_obs.new_zeros(()), pair_cycle_idx + 1

    # ── Compute learned s_θ WITH gradients ──
    # Get full rollout observation indices for both envs
    src_obs_indices = torch.arange(num_steps, device=device) * num_envs + src_env
    tgt_obs_indices = torch.arange(num_steps, device=device) * num_envs + tgt_env

    # Embeddings z_θ from actor trunk — WITH gradient!
    emb_x = agent.get_embedding(b_obs[src_obs_indices])  # (num_steps, 64)
    emb_y = agent.get_embedding(b_obs[tgt_obs_indices])  # (num_steps, 64)

    # Cosine similarity s_θ(x, y) — this is the learnable part
    emb_x_norm = emb_x / (emb_x.norm(dim=-1, keepdim=True) + 1e-8)
    emb_y_norm = emb_y / (emb_y.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_matrix = emb_x_norm @ emb_y_norm.T  # (num_steps, num_steps)

    # Γ from precomputed PSM — no gradient
    gamma_matrix = gamma_matrices[pair]  # (num_steps, num_steps)

    # Bidirectional contrastive loss
    loss = compute_psm_contrastive_loss(
        similarity_matrix, gamma_matrix, temperature
    )

    return loss, metric_means[pair], pair_cycle_idx + 1


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

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
            group=f"h_ppo_PSM_netembed_{args.group_name}"
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
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

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.n_keys, i, args.capture_video, run_name, args.random_color) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    episodes_returns = []
    episodes_lengths = []
    len_ep_ret = 0

    for iteration in trange(1, args.num_iterations + 1, colour="green"):
        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # ── Rollout phase ──
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

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

        # ── GAE advantage estimation ──
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

        # Flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ── Precompute PSM coupling weights Γ (fixed for this rollout) ──
        transition_terminals = build_transition_terminals(dones)
        gamma_matrices = None
        metric_means = None

        psm_loss_running = torch.tensor(0.0, device=device)
        psm_metric_running = torch.tensor(0.0, device=device)
        psm_count = 0
        psm_pair_cycle = 0

        if args.psm_aux_coef > 0:
            gamma_matrices, metric_means = precompute_psm_metrics(
                agent, b_obs,
                num_steps=args.num_steps,
                num_envs=args.num_envs,
                transition_terminals=transition_terminals,
                gamma=args.gamma,
                dp_eps=args.psm_dp_eps,
                beta=args.psm_beta,
            )

        # ── PPO update ──
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss (PPO clipped)
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

                # ── PSM contrastive auxiliary loss ──
                mb_psm_loss = torch.tensor(0.0, device=device)
                if args.psm_aux_coef > 0 and gamma_matrices is not None:
                    mb_psm_loss, mb_metric, psm_pair_cycle = compute_psm_loss_for_minibatch(
                        agent, b_obs, gamma_matrices, metric_means,
                        mb_inds, num_envs=args.num_envs,
                        num_steps=args.num_steps,
                        temperature=args.psm_temperature,
                        pair_cycle_idx=psm_pair_cycle,
                    )
                    psm_loss_running += mb_psm_loss.detach()
                    psm_metric_running += mb_metric.detach()
                    psm_count += 1

                # Total loss = PPO + α · PSM_contrastive
                loss = (pg_loss
                        - args.ent_coef * entropy_loss
                        + args.vf_coef * v_loss
                        + args.psm_aux_coef * mb_psm_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ── Logging ──
        avg_psm_loss = psm_loss_running / max(psm_count, 1)
        avg_psm_metric = psm_metric_running / max(psm_count, 1)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/psm_cme", avg_psm_loss.item(), global_step)
        writer.add_scalar("losses/psm_metric_mean", avg_psm_metric.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

    if args.save_model:
        os.makedirs(f"models/h_ppo_PSM_netembed_{args.size_env}x{args.size_env}_{args.n_keys}keys{args.run_code}", exist_ok=True)
        model_path = f"models/h_ppo_PSM_netembed_{args.size_env}x{args.size_env}_{args.n_keys}keys{args.run_code}/h_ppo_PSM_netembed_seed={args.seed}.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
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
                group_name=f"h_ppo_PSM_netembed_{args.group_name}_evals",
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

    envs.close()
    writer.close()
