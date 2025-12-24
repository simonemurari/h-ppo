import numpy as np
from minigrid.core.constants import IDX_TO_COLOR
import torch

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
            "pickup": 3,  # Pickup object
            "toggle": 5,  # Open door
        }


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