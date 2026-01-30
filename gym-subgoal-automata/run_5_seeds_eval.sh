# #!/bin/bash
    
master_seeds=(70 75 95 200 700)

# DeliverCoffee -> DeliverCoffeeAndMail

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "ppo" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "ppo" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "ppo" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "_theta_0.25" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "_theta_0.25" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "_theta_0.25" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "_theta_0.5" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "_theta_0.5" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "_theta_0.5" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "_theta_0.75" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "_theta_0.75" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "_theta_0.75" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track


# PatrolAB -> PatrolABC

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "h_ppo_product" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "ppo" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "ppo" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "ppo" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "h_ppo_symloss" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "h_ppo_symloss_eps" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_theta_0.25_v2" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_theta_0.25_v2" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_theta_0.25_v2" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_theta_0.5_v2" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_theta_0.5_v2" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_theta_0.5_v2" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track

uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_theta_0.75_v2" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 1.0 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_theta_0.75_v2" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.5 --no-track
uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_theta_0.75_v2" --model_name "h_ppo_symloss_theta" --eval_type "h_ppo" --epsilon 0.25 --no-track


# # ppo_reward_machine evals (both maps, all eval types)

# # DeliverCoffee -> DeliverCoffeeAndMail (ppo_reward_machine)
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "ppo_reward_machine" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "ppo_reward_machine" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 1.0 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 0.5 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "DeliverCoffee" --task "DeliverCoffeeAndMail" --run_code "" --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 0.25 --no-track

# # PatrolAB -> PatrolABC (ppo_reward_machine)
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "ppo_reward_machine" --eval_type "standard" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "ppo_reward_machine" --eval_type "random_rules" --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 1.0 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 0.5 --no-track
# uv run aggregate_eval.py --master_seeds "${master_seeds[@]}" --task_model "PatrolAB" --task "PatrolABC" --run_code "_v2" --model_name "ppo_reward_machine" --eval_type "h_ppo" --epsilon 0.25 --no-track