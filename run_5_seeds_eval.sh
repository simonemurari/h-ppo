# #!/bin/bash


# for seed in 50 70 95 200 700
# do
#     uv run ppo_eval.py --seed $seed --size_env 8 --run_code "RC" --n_keys 2
# done
# for seed in 50 70 95 200 700
# do
#     uv run ppo_eval.py --seed $seed --heuristic --size_env 8 --run_code "_ef0_RC" --n_keys 2
# done


# for seed in 50 70 95 200 700
# do
#     uv run ppo_eval.py --seed $seed --size_env 8 --run_code "RC" --n_keys 4
# done
# for seed in 50 70 95 200 700
# do
#     uv run ppo_eval.py --seed $seed --heuristic --size_env 8 --run_code "_ef0_RC" --n_keys 4
# done


# for seed in 50 70 95 200 700
# do
#     uv run ppo_eval.py --seed $seed --size_env 16 --run_code "RC"
# done
# for seed in 50 70 95 200 700
# do
#     uv run ppo_eval.py --seed $seed --heuristic --size_env 16 --run_code "_ef0_RC"
# done


# for seed in 50 70 95 200 700
# do
#     uv run ppo_eval.py --seed $seed --size_env 16 --run_code "RC" --n_keys 2
# done
# for seed in 50 70 95 200 700
# do
#     uv run ppo_eval.py --seed $seed --heuristic --size_env 16 --run_code "_ef0_RC" --n_keys 2
# done


# for seed in 50 70 95 200 700
# do
#     uv run ppo_eval.py --seed $seed --size_env 16 --run_code "RC" --n_keys 4
# done
for seed in 95 200 700
do
    uv run ppo_eval.py --seed $seed --heuristic --size_env 16 --run_code "_ef0_RC" --n_keys 4
done