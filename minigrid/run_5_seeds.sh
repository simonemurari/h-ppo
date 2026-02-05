#!/bin/bash

# 16x16 1 key (16 num_envs)
for seed in 70 50 95 200 700
do
    uv run ppo_reward_machine.py --seed $seed --size_env 8 --n_keys 1 --run_code "TEST"
done