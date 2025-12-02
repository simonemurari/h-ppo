#!/bin/bash

for seed in 50 70 95 200 700
do
    uv run h_ppo_product.py --seed $seed --run_code "TEST"
done