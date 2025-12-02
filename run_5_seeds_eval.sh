#!/bin/bash

for seed in 50 70 95 200 700
do
    uv run ppo_eval.py --seed $seed --run_code "TEST" --n_keys 2
done
for seed in 50 70 95 200 700
do
    uv run ppo_eval.py --seed $seed --run_code "TEST" --n_keys 4
done
for seed in 50 70 95 200 700
do
    uv run ppo_eval.py --seed $seed --run_code "TEST" --size_env 16
done
for seed in 50 70 95 200 700
do
    uv run ppo_eval.py --seed $seed --run_code "TEST" --size_env 16 --n_keys 2
done
for seed in 50 70 95 200 700
do
    uv run ppo_eval.py --seed $seed --run_code "TEST" --size_env 16 --n_keys 4
done