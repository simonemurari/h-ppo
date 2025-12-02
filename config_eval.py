
from dataclasses import dataclass
import os
from dotenv import load_dotenv
load_dotenv()
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME", "cleanRL")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class Args:
    seed: int = 21
    """seed of the experiment"""

    size_env: int = 8
    """the size of the environment (8, 16)"""

    size_env_model: int = 8
    """the size of the environment the model was trained on (8, 16)"""

    n_keys: int = 1
    """the number of keys in the environment"""

    n_keys_model: int = 1
    """the number of keys the model was trained on"""

    run_code: str = "TEST"
    """an optional code to distinguish the runs"""

    @property
    def env_id(self) -> str:
        """the id of the environment"""
        return f"MiniGrid-DoorKey-{self.size_env}x{self.size_env}-v0"

    heuristic: bool = False
    """whether to use the heuristic model or the standard PPO model"""

    eval_episodes: int = 1000
    """the number of evaluation episodes"""

    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    track: bool = True
    """whether to track the experiment with wandb"""

    wandb_project_name: str = WANDB_PROJECT_NAME
    """the wandb project name"""

    wandb_entity: str = WANDB_ENTITY
    """the wandb entity (team) name"""

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""

    @property
    def group_name(self) -> str:
        """the wandb's group name for the experiment"""
        if self.heuristic is True:
            return f"h_ppo_product_{self.size_env_model}x{self.size_env_model}_{self.n_keys_model}keys-TRAINED_{self.size_env}x{self.size_env}_{self.n_keys}keys{self.run_code}-EVAL"
        else:
            return f"ppo_{self.size_env_model}x{self.size_env_model}_{self.n_keys_model}keys-TRAINED_{self.size_env}x{self.size_env}_{self.n_keys}keys{self.run_code}-EVAL"

    @property
    def model_path(self) -> str:
        if self.heuristic is True:
            return f"models/h_ppo_product_{self.size_env_model}x{self.size_env_model}_{self.n_keys_model}keys{self.run_code}/h_ppo_seed={self.seed}.pt"
        else:
            return f"models/ppo_{self.size_env_model}x{self.size_env_model}_{self.n_keys_model}keys{self.run_code}/ppo_seed={self.seed}.pt"
    """the path to the saved model"""