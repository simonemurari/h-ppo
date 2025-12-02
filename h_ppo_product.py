# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from config import Args
import gymnasium as gym
import minigrid
from minigrid.core.constants import IDX_TO_COLOR
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


DOOR_STATES = ["open", "closed", "locked"]
VIEW_SIZE = 7
MID_POINT = (VIEW_SIZE - 1) // 2

OFFSETS_X, OFFSETS_Y = np.meshgrid(
    np.arange(VIEW_SIZE) - MID_POINT,
    np.abs(np.arange(VIEW_SIZE) - (VIEW_SIZE - 1)),
    indexing="ij",
)



def make_env(env_id, n_keys, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, n_keys=n_keys)
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
        self.action_map = {
            "left": 0,  # Turn left
            "right": 1,  # Turn right
            "forward": 2,  # Move forward
            "pickup": 3,  # Pickup object
            "toggle": 5,  # Open door
        }
        self.conf_level = 0.8

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, apply_heuristic=False, epsilon=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        
        if apply_heuristic is True:
            suggested_actions_batch = self._apply_rules_batch(self.get_observables(x[:, 4:]))
            multiplier = torch.ones_like(logits)
            for i in range(x.shape[0]):
                suggested_actions_list = suggested_actions_batch[i]
                for suggested_action in suggested_actions_list:
                    if suggested_action is not None:
                        multiplier[i, suggested_action] += self.conf_level * epsilon
            modified_probs = probs.probs * multiplier
            modified_probs = modified_probs / modified_probs.sum(dim=-1, keepdim=True)
            probs = Categorical(probs=modified_probs)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def _apply_rules_batch(self, batch_observables):
            """Apply rules to each environment observation in the batch"""
            batch_rule_actions = []
            
            for observables in batch_observables:
                # Parse observables
                keys = [o for o in observables if o[0] == "key"]
                doors = [o for o in observables if o[0] == "door"]
                goals = [o for o in observables if o[0] == "goal"]
                walls = [o for o in observables if o[0] == "wall"]
                carrying_keys = [o for o in observables if o[0] == "carryingKey"]
                locked_doors = [o for o in observables if o[0] == "locked"]
                closed_doors = [o for o in observables if o[0] == "closed"]
                rule_actions = []
    
                # Rule 1: pickup(X) :- key(X), samecolor(X,Y), door(Y), notcarrying
                if keys and doors and not carrying_keys:
                    for key in keys:
                        key_color = key[1][0]
                        matching_doors = [door for door in doors if door[1][0] == key_color]
                        if matching_doors:
                            # Check if key is directly in front
                            key_x, key_y = key[1][1], key[1][2]
                            if key_x == 0 and key_y == 1:  # Key is directly in front
                                rule_actions.append(self.action_map["pickup"])
                                break
                            else:
                                # Move towards the key with wall avoidance
                                action = self._navigate_towards(key_x, key_y, walls)
                                rule_actions.append(action)
                                break
    
                # Rule 2: open(X) :- door(X), locked(X), key(Z), carryingKey(Z), samecolor(X,Z)
                if doors and locked_doors and carrying_keys:
                    carrying_key_color = carrying_keys[0][1][0]
    
                    # Check locked doors first (priority)
                    matching_doors_to_open = []
                    if locked_doors:
                        for door in doors:
                            door_color = door[1][0]
                            if door_color == carrying_key_color:
                                for locked in locked_doors:
                                    if locked[1][0] == door_color:
                                        matching_doors_to_open.append(door)
    
                    if matching_doors_to_open:
                        door = matching_doors_to_open[0]
                        door_x, door_y = door[1][1], door[1][2]
                        if door_x == 0 and door_y == 1:  # Door is directly in front
                            rule_actions.append(self.action_map["toggle"])
                        else:
                            # Move towards the door with wall avoidance
                            action = self._navigate_towards(door_x, door_y, walls)
                            rule_actions.append(action)
    
                # Rule 3: goto :- goal(X), unlocked
                if goals:
                    goal = goals[0]
                    goal_x, goal_y = goal[1][0], goal[1][1]
    
                    # Check if there's a clear path to the goal (no closed/locked doors in the way)
                    blocked_by_door = False
    
                    # Simple check: if we see a closed/locked door that's between us and the goal
                    direction_to_goal = (
                        1 if goal_x > 0 else (-1 if goal_x < 0 else 0),
                        1 if goal_y > 0 else (-1 if goal_y < 0 else 0),
                    )
    
                    # Only consider a door blocking if it's in the same general direction as the goal
                    for door in doors:
                        door_x, door_y = door[1][1], door[1][2]
                        door_direction = (
                            1 if door_x > 0 else (-1 if door_x < 0 else 0),
                            1 if door_y > 0 else (-1 if door_y < 0 else 0),
                        )
    
                        door_color = door[1][0]
                        # Check if the door is in the same general direction as the goal
                        same_direction = (
                            direction_to_goal[0] == door_direction[0]
                            and direction_to_goal[1] == door_direction[1]
                        )
    
                        # Check if door is closer than the goal
                        door_distance = abs(door_x) + abs(door_y)
                        goal_distance = abs(goal_x) + abs(goal_y)
                        door_is_closer = door_distance < goal_distance
    
                        # Check if the door is closed or locked
                        door_is_closed = any(cd[1][0] == door_color for cd in closed_doors)
                        door_is_locked = any(ld[1][0] == door_color for ld in locked_doors)
    
                        if (
                            same_direction
                            and door_is_closer
                            and (door_is_closed or door_is_locked)
                        ):
                            blocked_by_door = True
                            break
    
                    if not blocked_by_door:
                        if goal_x == 0 and goal_y == 1:  # Goal is directly in front
                            rule_actions.append(self.action_map["forward"])
                        else:
                            # Move towards the goal with wall avoidance
                            action = self._navigate_towards(goal_x, goal_y, walls)
                            rule_actions.append(action)
                            
                if len(rule_actions) == 0:
                    rule_actions.append(None)
                batch_rule_actions.append(rule_actions)
    
            return batch_rule_actions
    
    def _navigate_towards(self, target_x, target_y, walls=None):
        """
        Improved navigation helper that avoids walls when moving towards a target
    
        Args:   
            target_x: Relative x-coordinate of the target
            target_y: Relative y-coordinate of the target
            walls: List of wall observations with their positions
        """
        # If no walls, use simpler navigation
        if not walls:
            if target_y > 0:  # Target is in front
                return self.action_map["forward"]
            elif target_x < 0:  # Target is to the left
                return self.action_map["left"]
            elif target_x > 0:  # Target is to the right
                return self.action_map["right"]
            else:  # Target is behind, turn around
                return self.action_map["right"]
    
        # Check if there's a wall directly in front
        wall_in_front = any(w[1][0] == 0 and w[1][1] == 1 for w in walls)
    
        # Determine the relative position of the target
        if target_y > 0:  # Target is in front
            if not wall_in_front:
                return self.action_map["forward"]
            else:
                # Wall blocking forward movement, turn to find another path
                return (
                    self.action_map["left"]
                    if target_x <= 0
                    else self.action_map["right"]
                )
        elif target_x < 0:  # Target is to the left
            return self.action_map["left"]
        elif target_x > 0:  # Target is to the right
            return self.action_map["right"]
        else:  # Target is behind
            # Choose a turn direction based on wall presence
            wall_to_left = any(w[1][0] == -1 and w[1][1] == 0 for w in walls)
            if wall_to_left:
                return self.action_map["right"]
            else:
                return self.action_map["left"]
    
    # Optimize observation processing with NumPy
    def get_observables(self, raw_obs_batch):
        """
        Highly optimized version of get_observables that processes entire batch at once
        """
        batch_size = raw_obs_batch.shape[0]
    
        # Convert to NumPy once if needed
        if isinstance(raw_obs_batch, torch.Tensor):
            raw_obs_batch = raw_obs_batch.cpu().numpy()
    
        # Reshape efficiently with pre-computed shape
        try:
            img_batch = raw_obs_batch.reshape(batch_size, VIEW_SIZE, VIEW_SIZE, 3)
        except ValueError:
            # Handle case where dimensions don't match by taking only the image part
            img_batch = raw_obs_batch[:, : VIEW_SIZE * VIEW_SIZE * 3].reshape(
                batch_size, VIEW_SIZE, VIEW_SIZE, 3
            )
    
        # Process batch items in parallel
        batch_obs = []
    
        # Process each batch item with minimal Python overhead
        for img in img_batch:
            obs = []
            item_first = img[..., 0]
            item_second = img[..., 1]
            item_third = img[..., 2]
    
            # Find all object positions efficiently with NumPy
            key_positions = np.where(item_first == 5)
            door_positions = np.where(item_first == 4)
            goal_positions = np.where(item_first == 8)
            wall_positions = np.where(item_first == 2)
    
            # Vectorized processing for keys
            for k_i, k_j in zip(*key_positions):
                color = IDX_TO_COLOR.get(item_second[k_i, k_j])
                obs.append(("key", [color, OFFSETS_X[k_i, k_j], OFFSETS_Y[k_i, k_j]]))
                if k_i == MID_POINT and k_j == VIEW_SIZE - 1:
                    obs.append(("carryingKey", [color]))
    
            # Vectorized processing for doors
            for d_i, d_j in zip(*door_positions):
                color = IDX_TO_COLOR.get(item_second[d_i, d_j])
                obs.append(("door", [color, OFFSETS_X[d_i, d_j], OFFSETS_Y[d_i, d_j]]))
                # Get the door state from the third channel
                door_state_idx = int(item_third[d_i, d_j])
                obs.append((DOOR_STATES[door_state_idx], [color]))
    
            # Vectorized processing for goals and walls
            for g_i, g_j in zip(*goal_positions):
                obs.append(("goal", [OFFSETS_X[g_i, g_j], OFFSETS_Y[g_i, g_j]]))
    
            for w_i, w_j in zip(*wall_positions):
                obs.append(("wall", [OFFSETS_X[w_i, w_j], OFFSETS_Y[w_i, w_j]]))
    
            batch_obs.append(obs)
    
        return batch_obs
    
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

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
            group=f"h_ppo_product_{args.group_name}"
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
        [make_env(args.env_id, args.n_keys, i, args.capture_video, run_name) for i in range(args.num_envs)],
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

        epsilon = linear_schedule(
                    args.start_e,
                    args.end_e,
                    args.exploration_fraction * args.total_timesteps,
                    global_step,
                )

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, apply_heuristic=True, epsilon=epsilon)
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
                        if iteration % 100 == 0:
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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
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
        os.makedirs(f"models/h_ppo_product_{args.size_env}x{args.size_env}_{args.n_keys}keys{args.run_code}", exist_ok=True)
        model_path = f"models/h_ppo_product_{args.size_env}x{args.size_env}_{args.n_keys}keys{args.run_code}/h_ppo_seed={args.seed}.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            args.n_keys,
            eval_episodes=1000,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            track=args.track
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # if args.upload_model:
        #     from cleanrl_utils.huggingface import push_to_hub

        #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
        #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #     push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()