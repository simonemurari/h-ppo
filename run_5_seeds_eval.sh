#!/bin/bash

# epsilon 0.25 _ef_0_RC

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 2 --eval_code "_H_PPO" --epsilon 0.25 --heuristic
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 4 --eval_code "_H_PPO" --epsilon 0.25 --heuristic
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 1 --size_env 16 --eval_code "_H_PPO" --epsilon 0.25 --heuristic
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 2 --size_env 16 --eval_code "_H_PPO" --epsilon 0.25 --heuristic
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 4 --size_env 16 --eval_code "_H_PPO" --epsilon 0.25 --heuristic
done

# epsilon 0.1 _ef_0_RC

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 2 --eval_code "_H_PPO" --epsilon 0.1 --heuristic
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 4 --eval_code "_H_PPO" --epsilon 0.1 --heuristic
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 1 --size_env 16 --eval_code "_H_PPO" --epsilon 0.1 --heuristic
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 2 --size_env 16 --eval_code "_H_PPO" --epsilon 0.1 --heuristic
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 4 --size_env 16 --eval_code "_H_PPO" --epsilon 0.1 --heuristic
done

# epsilon 0.05 _ef_0_RC

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 2 --eval_code "_H_PPO" --epsilon 0.05 --heuristic
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 4 --eval_code "_H_PPO" --epsilon 0.05 --heuristic
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 1 --size_env 16 --eval_code "_H_PPO" --epsilon 0.05 --heuristic
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 2 --size_env 16 --eval_code "_H_PPO" --epsilon 0.05 --heuristic
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 4 --size_env 16 --eval_code "_H_PPO" --epsilon 0.05 --heuristic
done

# epsilon 0.01 _ef_0_RC

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 2 --eval_code "_H_PPO" --epsilon 0.01 --heuristic
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 4 --eval_code "_H_PPO" --epsilon 0.01 --heuristic
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 1 --size_env 16 --eval_code "_H_PPO" --epsilon 0.01 --heuristic
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 2 --size_env 16 --eval_code "_H_PPO" --epsilon 0.01 --heuristic
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "_ef_0_RC" --n_keys 4 --size_env 16 --eval_code "_H_PPO" --epsilon 0.01 --heuristic
done

# epsilon 0.01 RC

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "RC" --n_keys 2 --eval_code "_H_PPO" --epsilon 0.01
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "RC" --n_keys 4 --eval_code "_H_PPO" --epsilon 0.01
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "RC" --n_keys 1 --size_env 16 --eval_code "_H_PPO" --epsilon 0.01
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "RC" --n_keys 2 --size_env 16 --eval_code "_H_PPO" --epsilon 0.01
done

for seed in 95 200 700 50 70
do
    uv run h_ppo_eval.py --seed $seed --run_code "RC" --n_keys 4 --size_env 16 --eval_code "_H_PPO" --epsilon 0.01
done