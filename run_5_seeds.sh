#!/bin/bash

for seed in 50 70 95 200 700
do
    uv run h_ppo_product.py --seed $seed --run_code "_ef0_RC_1" --end_e 0 --save_model
done

