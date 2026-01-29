# #!/bin/bash
    
master_seeds=(70 75 95 200 700)

# DeliverCoffee -> DeliverCoffeeAndMail
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 0.5 --no-track

# PatrolAB -> PatrolABC  
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 0.5 --no-track