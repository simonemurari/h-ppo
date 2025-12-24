#!/bin/bash

for seed in 700 200 95 50 70
do
    uv run ppo.py --seed $seed --run_code "RC" --save_model
done

for seed in 50 70 95 200 700
do
    uv run h_ppo_product.py --seed $seed --run_code "_ef0_RC" --end_e 0 --save_model
done

