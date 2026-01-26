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
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""

    seed: int = 50
    """seed of the experiment"""

    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""

    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""

    wandb_project_name: str = WANDB_PROJECT_NAME
    """the wandb's project name"""

    wandb_entity: str = WANDB_ENTITY
    """the entity (team) of wandb's project"""

    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    save_model: bool = False
    """whether to save the model into a .pt file"""

    # Algorithm specific arguments
    task: str = "DeliverCoffee"  # DeliverCoffee, DeliverCoffeeAndMail, PatrolAB, PatrolABC
    """the task to run the experiments on"""

    run_code: str = ""
    """an optional code to distinguish the runs"""

    @property
    def env_id(self) -> str:
        """the id of the environment"""
        return f"gym_subgoal_automata:OfficeWorld{self.task}-v0"
    
    @property
    def group_name(self) -> str:
        """the wandb's group name for the experiment"""
        if self.run_code != "" and not self.run_code.startswith("_"):
            self.run_code = "_" + self.run_code

        return f"{self.task}{self.run_code}"

    total_timesteps: int = 500_000
    """total timesteps of the experiments"""

    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""

    num_envs: int = 4
    """the number of parallel game environments"""

    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""

    gamma: float = 0.99
    """the discount factor gamma"""

    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""

    num_minibatches: int = 4
    """the number of mini-batches"""

    update_epochs: int = 4
    """the K epochs to update the policy"""

    norm_adv: bool = True
    """Toggles advantages normalization"""

    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""

    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""

    ent_coef: float = 0.01
    """coefficient of the entropy"""

    vf_coef: float = 0.5
    """coefficient of the value function"""

    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    target_kl: float = None
    """the target KL divergence threshold"""

    start_e: float = 1.0
    """starting value for heuristic epsilon schedule"""

    end_e: float = 0
    """final value for heuristic epsilon schedule"""

    exploration_fraction: float = 0.4
    """fraction of total iterations over which epsilon decays linearly"""

    theta: float = 0.75
    """coefficient for symbolic loss term"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""

    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""

    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
