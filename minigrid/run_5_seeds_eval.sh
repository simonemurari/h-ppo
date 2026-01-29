#!/bin/bash

master_seeds=(50 70 95 200 700)

# ###############################################################################
# # 8x8 - 2 KEYS
# ###############################################################################

# # --- Standard Evaluation ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_eps" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 8 --model_name "h_ppo_product" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 8 --model_name "ppo" --eval_type "standard" --no-track

# # --- H-PPO Evaluation (h_ppo_product only) ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 8 --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 1.0 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 8 --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 0.5 --no-track

# # --- Random Rules Evaluation ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss_eps" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 8 --model_name "h_ppo_symloss" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 8 --model_name "h_ppo_product" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 8 --model_name "ppo" --eval_type "random_rules" --no-track

# ###############################################################################
# # 8x8 - 4 KEYS
# ###############################################################################

# # --- Standard Evaluation ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_eps" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 8 --model_name "h_ppo_product" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 8 --model_name "ppo" --eval_type "standard" --no-track

# # --- H-PPO Evaluation (h_ppo_product only) ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 8 --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 1.0 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 8 --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 0.5 --no-track

# # --- Random Rules Evaluation ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_theta" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss_eps" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 8 --model_name "h_ppo_symloss" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 8 --model_name "h_ppo_product" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 8 --model_name "ppo" --eval_type "random_rules" --no-track

# ###############################################################################
# # 16x16 - 1 KEY
# ###############################################################################

# # --- Standard Evaluation ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_eps" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 1 --size_env 16 --model_name "h_ppo_product" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 1 --size_env 16 --model_name "ppo" --eval_type "standard" --no-track

# # --- H-PPO Evaluation (h_ppo_product only) ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 1 --size_env 16 --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 1.0 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 1 --size_env 16 --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 0.5 --no-track

# # --- Random Rules Evaluation ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss_eps" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 1 --size_env 16 --model_name "h_ppo_symloss" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 1 --size_env 16 --model_name "h_ppo_product" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 1 --size_env 16 --model_name "ppo" --eval_type "random_rules" --no-track

# ###############################################################################
# # 16x16 - 2 KEYS
# ###############################################################################

# # --- Standard Evaluation ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_eps" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 16 --model_name "h_ppo_product" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 16 --model_name "ppo" --eval_type "standard" --no-track

# # --- H-PPO Evaluation (h_ppo_product only) ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 16 --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 1.0 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 16 --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 0.5 --no-track

# # --- Random Rules Evaluation ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss_eps" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 2 --size_env 16 --model_name "h_ppo_symloss" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 16 --model_name "h_ppo_product" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 2 --size_env 16 --model_name "ppo" --eval_type "random_rules" --no-track

# ###############################################################################
# # 16x16 - 4 KEYS
# ###############################################################################

# # --- Standard Evaluation ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "standard" --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "standard" --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_eps" --eval_type "standard" --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss" --eval_type "standard" --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 16 --model_name "h_ppo_product" --eval_type "standard" --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 16 --model_name "ppo" --eval_type "standard" --no-track

# --- H-PPO Evaluation (h_ppo_product only) ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 16 --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 1.0 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 16 --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 0.5 --no-track

# --- Random Rules Evaluation ---
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.25" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.5" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_theta_0.75" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_theta" --eval_type "random_rules" --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss_eps" --eval_type "random_rules" --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --n_keys 4 --size_env 16 --model_name "h_ppo_symloss" --eval_type "random_rules" --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 16 --model_name "h_ppo_product" --eval_type "random_rules" --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --run_code "_RC_v7" --n_keys 4 --size_env 16 --model_name "ppo" --eval_type "random_rules" --no-track

###############################################################################
# Generate consolidated plots
###############################################################################
# uv run aggregate_evals/plot_results.py
