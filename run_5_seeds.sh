#!/bin/bash

# for seed in 50 70 95 200 700
# do
#     uv run h_ppo_product.py --seed $seed --n_keys 1 --end_e 0.05 --run_code "ef_005"
# done

# for seed in 50 70 95 200 700
# do
#     uv run ppo.py --seed $seed --total_timesteps 10000000 --run_code "10M_steps" --num_envs 8
# done

for seed in 50 70 95 200 700
do
    uv run h_ppo_product.py --seed $seed --total_timesteps 10000000 --run_code "10M_steps" --num_envs 8 --end_e 0.1
done