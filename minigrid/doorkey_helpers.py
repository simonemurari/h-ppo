import numpy as np
from minigrid.core.constants import IDX_TO_COLOR
import torch
import gymnasium as gym

DOOR_STATES = ["open", "closed", "locked"]
VIEW_SIZE = 7
MID_POINT = (VIEW_SIZE - 1) // 2

OFFSETS_X, OFFSETS_Y = np.meshgrid(
    np.arange(VIEW_SIZE) - MID_POINT,
    np.abs(np.arange(VIEW_SIZE) - (VIEW_SIZE - 1)),
    indexing="ij",
)

action_map = {
            "left": 0,  # Turn left
            "right": 1,  # Turn right
            "forward": 2,  # Move forward
            "pickup": 3,  # Pickup key
            "toggle": 5,  # Open/Unlock door
        }

COLOR_ORDER = [IDX_TO_COLOR[idx] for idx in sorted(IDX_TO_COLOR.keys())]
COLOR_TO_SLOT = {color: idx for idx, color in enumerate(COLOR_ORDER)}
GROUNDED_VECTOR_DIM = 64
MAX_DX = MID_POINT
MAX_DY = VIEW_SIZE - 1


def apply_rules_batch(batch_observables):
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
                        rule_actions.append(action_map["pickup"])
                        break
                    else:
                        # Move towards the key with wall avoidance
                        action = _navigate_towards(key_x, key_y, walls)
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
                    rule_actions.append(action_map["toggle"])
                else:
                    # Move towards the door with wall avoidance
                    action = _navigate_towards(door_x, door_y, walls)
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
                    rule_actions.append(action_map["forward"])
                else:
                    # Move towards the goal with wall avoidance
                    action = _navigate_towards(goal_x, goal_y, walls)
                    rule_actions.append(action)
                    
        if len(rule_actions) == 0:
            rule_actions.append(None)
        batch_rule_actions.append(rule_actions)

    return batch_rule_actions
    
def _navigate_towards(target_x, target_y, walls=None):
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
            return action_map["forward"]
        elif target_x < 0:  # Target is to the left
            return action_map["left"]
        elif target_x > 0:  # Target is to the right
            return action_map["right"]
        else:  # Target is behind, turn around
            return action_map["right"]

    # Check if there's a wall directly in front
    wall_in_front = any(w[1][0] == 0 and w[1][1] == 1 for w in walls)

    # Determine the relative position of the target
    if target_y > 0:  # Target is in front
        if not wall_in_front:
            return action_map["forward"]
        else:
            # Wall blocking forward movement, turn to find another path
            return (
                action_map["left"]
                if target_x <= 0
                else action_map["right"]
            )
    elif target_x < 0:  # Target is to the left
        return action_map["left"]
    elif target_x > 0:  # Target is to the right
        return action_map["right"]
    else:  # Target is behind
        # Choose a turn direction based on wall presence
        wall_to_left = any(w[1][0] == -1 and w[1][1] == 0 for w in walls)
        if wall_to_left:
            return action_map["right"]
        else:
            return action_map["left"]

# Optimize observation processing with NumPy
def get_observables(raw_obs_batch):
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


def _normalize_dx(dx):
    return float(np.clip(dx / max(1, MAX_DX), -1.0, 1.0))


def _normalize_dy(dy):
    return float(np.clip(dy / max(1, MAX_DY), -1.0, 1.0))


def _nearest_entity_by_color(entities, color):
    candidates = []
    for entity in entities:
        attrs = entity[1]
        if attrs[0] == color and len(attrs) >= 3:
            candidates.append((attrs[1], attrs[2]))
    if not candidates:
        return None
    return min(candidates, key=lambda pos: abs(pos[0]) + abs(pos[1]))


def observables_to_grounded_vectors(batch_observables):
    """Converts parsed observables to fixed 64-D grounded predicate vectors.

    Vector layout:
    - [0:6] key presence by color
    - [6:12] key dx by color
    - [12:18] key dy by color
    - [18:24] carrying-key color one-hot
    - [24:30] door presence by color
    - [30:36] door locked by color
    - [36:42] door closed by color
    - [42:48] door dx by color
    - [48:54] door dy by color
    - [54] goal present
    - [55] goal dx
    - [56] goal dy
    - [57] wall directly in front
    - [58] wall directly left
    - [59] wall directly right
    - [60] normalized wall count
    - [61] any key signal present
    - [62] any door signal present
    - [63] bias term
    """
    vectors = np.zeros((len(batch_observables), GROUNDED_VECTOR_DIM), dtype=np.float32)

    for idx, observables in enumerate(batch_observables):
        keys = [o for o in observables if o[0] == "key"]
        doors = [o for o in observables if o[0] == "door"]
        goals = [o for o in observables if o[0] == "goal"]
        walls = [o for o in observables if o[0] == "wall"]
        carrying_keys = [o for o in observables if o[0] == "carryingKey"]
        locked_doors = [o for o in observables if o[0] == "locked"]
        closed_doors = [o for o in observables if o[0] == "closed"]

        for color, slot in COLOR_TO_SLOT.items():
            nearest_key = _nearest_entity_by_color(keys, color)
            if nearest_key is not None:
                key_dx, key_dy = nearest_key
                vectors[idx, slot] = 1.0
                vectors[idx, 6 + slot] = _normalize_dx(key_dx)
                vectors[idx, 12 + slot] = _normalize_dy(key_dy)

            nearest_door = _nearest_entity_by_color(doors, color)
            if nearest_door is not None:
                door_dx, door_dy = nearest_door
                vectors[idx, 24 + slot] = 1.0
                vectors[idx, 42 + slot] = _normalize_dx(door_dx)
                vectors[idx, 48 + slot] = _normalize_dy(door_dy)

        for carrying in carrying_keys:
            carrying_color = carrying[1][0]
            if carrying_color in COLOR_TO_SLOT:
                vectors[idx, 18 + COLOR_TO_SLOT[carrying_color]] = 1.0

        for locked in locked_doors:
            locked_color = locked[1][0]
            if locked_color in COLOR_TO_SLOT:
                vectors[idx, 30 + COLOR_TO_SLOT[locked_color]] = 1.0

        for closed in closed_doors:
            closed_color = closed[1][0]
            if closed_color in COLOR_TO_SLOT:
                vectors[idx, 36 + COLOR_TO_SLOT[closed_color]] = 1.0

        if goals:
            nearest_goal = min(goals, key=lambda goal: abs(goal[1][0]) + abs(goal[1][1]))
            vectors[idx, 54] = 1.0
            vectors[idx, 55] = _normalize_dx(nearest_goal[1][0])
            vectors[idx, 56] = _normalize_dy(nearest_goal[1][1])

        wall_positions = {(wall[1][0], wall[1][1]) for wall in walls if len(wall[1]) >= 2}
        vectors[idx, 57] = 1.0 if (0, 1) in wall_positions else 0.0
        vectors[idx, 58] = 1.0 if (-1, 0) in wall_positions else 0.0
        vectors[idx, 59] = 1.0 if (1, 0) in wall_positions else 0.0
        vectors[idx, 60] = min(len(wall_positions) / float(VIEW_SIZE * VIEW_SIZE), 1.0)

        vectors[idx, 61] = 1.0 if (len(keys) > 0 or len(carrying_keys) > 0) else 0.0
        vectors[idx, 62] = 1.0 if len(doors) > 0 else 0.0
        vectors[idx, 63] = 1.0

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-8)
    return vectors.astype(np.float32)


def get_grounded_predicate_vectors(raw_obs_batch):
    """Batched wrapper: raw observations -> observables -> grounded vectors."""
    return observables_to_grounded_vectors(get_observables(raw_obs_batch))


class MixedKeyEnvWrapper(gym.Wrapper):
    """Randomly samples n_keys ∈ {1, 2} at each episode reset.

    Useful for mixed training: with probability p_2key the episode uses 2 keys
    (one matching the door, one distractor), otherwise 1 key as usual.
    The observation and action spaces are identical in both cases, so this is
    a transparent drop-in: just wrap the env in make_env when p_2key > 0.

    Args:
        env_id:       gymnasium env ID (e.g. 'MiniGrid-DoorKey-8x8-v0')
        p_2key:       probability of using n_keys=2 at each reset (default 0.1)
        random_color: whether to randomize key/door color (same as DoorKeyEnv)
    """

    def __init__(self, env_id: str, p_2key: float = 0.1, random_color: bool = True):
        self._env_id = env_id
        self._p_2key = p_2key
        self._random_color = random_color
        # Build initial env with n_keys=1 to initialise parent
        env = self._build_inner(n_keys=1)
        super().__init__(env)

    def _build_inner(self, n_keys: int) -> gym.Env:
        """Creates a fresh FlattenObservation(FilterObservation(DoorKeyEnv)) stack."""
        env = gym.make(self._env_id, n_keys=n_keys, random_color=self._random_color)
        env = gym.wrappers.FlattenObservation(
            gym.wrappers.FilterObservation(env, filter_keys=["image", "direction"])
        )
        return env

    def reset(self, **kwargs):
        # Sample n_keys for this episode
        n_keys = 2 if np.random.random() < self._p_2key else 1
        self.env = self._build_inner(n_keys)
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)