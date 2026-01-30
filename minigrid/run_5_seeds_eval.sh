#!/bin/bash

master_seeds=(50 70 95 200 700)

###############################################################################
# 8x8 - 2 KEYS (H-PPO Evaluation)
###############################################################################

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 8 --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 8 --model_name "ppo" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 8 --model_name "ppo" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 8 --model_name "ppo" --eval_type "h_ppo" --epsilon 0.25 --no-track

###############################################################################
# 8x8 - 4 KEYS (H-PPO Evaluation)
###############################################################################

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 8 --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 8 --model_name "ppo" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 8 --model_name "ppo" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 8 --model_name "ppo" --eval_type "h_ppo" --epsilon 0.25 --no-track

###############################################################################
# 16x16 - 1 KEY (H-PPO Evaluation)
###############################################################################

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 1 --size_env 16 --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 1 --size_env 16 --model_name "ppo" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 1 --size_env 16 --model_name "ppo" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 1 --size_env 16 --model_name "ppo" --eval_type "h_ppo" --epsilon 0.25 --no-track

###############################################################################
# 16x16 - 2 KEYS (H-PPO Evaluation)
###############################################################################

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 16 --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 16 --model_name "ppo" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 16 --model_name "ppo" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 16 --model_name "ppo" --eval_type "h_ppo" --epsilon 0.25 --no-track

###############################################################################
# 16x16 - 4 KEYS (H-PPO Evaluation)
###############################################################################

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 16 --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 16 --model_name "ppo" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 16 --model_name "ppo" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 16 --model_name "ppo" --eval_type "h_ppo" --epsilon 0.25 --no-track


# ###############################################################################
# # ppo_reward_machine evals (all maps, all eval types)
# ###############################################################################

# # --- 8x8 - 2 KEYS (ppo_reward_machine) ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 8 --model_name "ppo_reward_machine" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 8 --model_name "ppo_reward_machine" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 8 --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 1.0 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 8 --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 0.5 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 8 --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 0.25 --no-track

# # --- 8x8 - 4 KEYS (ppo_reward_machine) ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 8 --model_name "ppo_reward_machine" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 8 --model_name "ppo_reward_machine" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 8 --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 1.0 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 8 --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 0.5 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 8 --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 0.25 --no-track

# # --- 16x16 - 1 KEY (ppo_reward_machine) ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 1 --size_env 16 --model_name "ppo_reward_machine" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 1 --size_env 16 --model_name "ppo_reward_machine" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 1 --size_env 16 --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 1.0 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 1 --size_env 16 --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 0.5 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 1 --size_env 16 --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 0.25 --no-track

# # --- 16x16 - 2 KEYS (ppo_reward_machine) ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 16 --model_name "ppo_reward_machine" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 16 --model_name "ppo_reward_machine" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 16 --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 1.0 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 16 --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 0.5 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 16 --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 0.25 --no-track

# # --- 16x16 - 4 KEYS (ppo_reward_machine) ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 16 --model_name "ppo_reward_machine" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 16 --model_name "ppo_reward_machine" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 16 --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 1.0 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 16 --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 0.5 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 16 --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 0.25 --no-track

###############################################################################
# Generate consolidated plots
###############################################################################
# uv run aggregate_evals/plot_results.py
