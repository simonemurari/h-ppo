
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv
from typing import List
load_dotenv()
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME", "cleanRL")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class Args:
    seed: int = 50
    """seed of the experiment"""

    master_seeds: List[int] = field(default_factory=lambda: [50, 51, 52, 53, 54])
    """list of master seeds for aggregate evaluation"""

    size_env: int = 8
    """the size of the environment (8, 16)"""

    size_env_model: int = 8
    """the size of the environment the model was trained on (8, 16)"""

    n_keys: int = 1
    """the number of keys in the environment"""

    n_keys_model: int = 1
    """the number of keys the model was trained on"""

    run_code: str = ""
    """the same code used during training to distinguish different runs"""

    eval_code: str = ""
    """an optional code to distinguish the evaluation runs"""

    eval_type: str = "standard"
    """the type of evaluation: standard or random_rules or h_ppo"""

    random_color:  bool = True
    """whether to use a random color for the key in the 1 key environments instead of the default yellow color"""

    @property
    def env_id(self) -> str:
        """the id of the environment"""
        return f"MiniGrid-DoorKey-{self.size_env}x{self.size_env}-v0"

    epsilon: float | None = None
    """the epsilon value for the eval with h_ppo_eval.py"""

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

    model_name: str = "ppo"
    """the name of the model"""

    @property
    def group_name(self) -> str:
        """the wandb's group name for the experiment"""

        if not self.random_color and self.n_keys == 1:
            raise ValueError("random_color can only be False when n_keys is 1 because with more than 1 key, the colors are always random.")

        if self.run_code != "" and not self.run_code.startswith("_"):
            self.run_code = "_" + self.run_code

        if self.epsilon is None:
            return f"{self.model_name}_{self.size_env_model}x{self.size_env_model}_{self.n_keys_model}keys{self.run_code}-TRAINED_{self.size_env}x{self.size_env}_{self.n_keys}keys{self.eval_code}-EVAL"
        else:
            return f"{self.model_name}_{self.size_env_model}x{self.size_env_model}_{self.n_keys_model}keys{self.run_code}-TRAINED_{self.size_env}x{self.size_env}_{self.n_keys}keys{self.eval_code}-EPS={self.epsilon}-EVAL"

    @property
    def model_path(self) -> str:
        """the path to the model checkpoint"""
        return f"models/{self.model_name}_{self.size_env_model}x{self.size_env_model}_{self.n_keys_model}keys{self.run_code}/{self.model_name}_seed={self.seed}.pt"