#!/bin/bash


# for seed in 50 70 95 200 700
# do
#     uv run h_ppo_symloss_kl.py --seed $seed --n_keys 2 --num_envs 8 --total_timesteps 25_000_000 --no-save_model
# done

for seed in 50 70 95 200 700
do
    uv run h_ppo_symloss_theta.py --seed $seed --n_keys 4 --num_envs 16 --no-save_model --theta 0.25 --run_code "theta_0.25" --total_timesteps 50_000_000
done


for seed in 50 70 95 200 700
do
    uv run h_ppo_symloss_theta.py --seed $seed --n_keys 2 --size_env 16 --num_envs 32 --no-save_model --theta 0.75 --run_code "theta_0.75" --total_timesteps 25_000_000
done

for seed in 50 70 95 200 700
do
    uv run h_ppo_symloss_theta.py --seed $seed --n_keys 2 --size_env 16 --num_envs 32 --no-save_model --theta 0.5 --run_code "theta_0.5" --total_timesteps 25_000_000
done

for seed in 50 70 95 200 700
do
    uv run h_ppo_symloss_theta.py --seed $seed --n_keys 2 --size_env 16 --num_envs 32 --no-save_model --theta 0.25 --run_code "theta_0.25" --total_timesteps 25_000_000
done

for seed in 50
do
    uv run h_ppo_symloss_theta.py --seed $seed --n_keys 4 --size_env 16 --num_envs 64 --no-save_model --theta 0.75 --run_code "theta_0.75" --total_timesteps 100_000_000
done