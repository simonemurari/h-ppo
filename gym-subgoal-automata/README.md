## OfficeWorld and WaterWorld environments

In this repository you can find the code to run the PPO algorithm and its neurosymbolic variants on the OfficeWorld and WaterWorld environments. The code is adapted from CleanRL. In `env_README.md` you can find the original README.md of the gym-subgoal-automata repository and in `gym_subgoal_automata/` you can find the code for all the environments.

### Repository Structure

- **Baseline**:
  `ppo.py` is the standard PPO algorithm and `ppo_reward_machine.py` is the reward machine variant.

- **Neurosymbolic variants**:
  `h_ppo_product.py`, `h_ppo_symloss.py`, `h_ppo_symloss_eps.py`, `h_ppo_symloss_theta.py` are the neurosymbolic variants of PPO that integrate symbolic guidance into the learning process.

- **config.py**:
  This file is used to set the different parameters for each run (learning rate, batch size, environment settings, etc.).

### How to run

1. **Create a Python environment**
   Create a virtual environment with Python 3.7 and install the packages. For example, using conda:
   ```bash
   conda create -n gym-subgoal-automata python=3.7.9
   conda activate gym-subgoal-automata
   pip install -r requirements.txt
   ```
   Or you can install `uv` and set up the environment. Note that newer versions of `uv` do not support Python 3.7, so you should use an older version to install it through uvx:
   ```bash
   uvx uv@0.6.17 python install 3.7
   uv sync
   uv pip install -r requirements.txt
   ```

2. **Configure wandb (optional)** \
   If you want to use wandb tracking, create a `.env` file in this folder and set the proper environment variables (WANDB_KEY, WANDB_PROJECT_NAME, WANDB_ENTITY).

3. **Configure the config.py file** \
   You can find the hyperparameters to configure the `config.py` file to replicate the experiments of the paper in each taask looking at the Appendix in the paper and the seeds tested in `seeds.txt`

4. **Run the scripts** \
   To run the different scripts just do e.g.: ```python h_ppo_product.py``` or ```uv run h_ppo_product.py``` if you are using `uv`
