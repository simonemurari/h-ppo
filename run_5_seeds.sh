#!/bin/bash

for seed in 50 70 95 200 700
do
    uv run ppo.py --seed $seed --run_code "TEST"
done