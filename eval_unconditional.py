import os
import sys
import uuid
import glob
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import wandb
import numpy as np


# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
ddp_rank = 0
ddp_world_size = 1
device = f'cuda:{ddp_rank}'
print(f"using device: {device}")
#master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.
master_process = True
# convenience variables

from omegaconf import OmegaConf


def load_config(experiment_config_path: str) -> OmegaConf:
    """Load and merge default config with experiment config."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, "configs", "default_config.yaml")
    
    if os.path.exists(default_config_path):
        default_config = OmegaConf.load(default_config_path)
    else:
        default_config = OmegaConf.create()
    
    experiment_config = OmegaConf.load(experiment_config_path)
    config = OmegaConf.merge(default_config, experiment_config)
    
    return config


config = load_config(sys.argv[1])

from Eval_utils import EvalMetric

metric = EvalMetric(device="cuda",max_length=config.data.sequence_length)

from tqdm import tqdm
B = 1

res = {'GPT2': [],
     'GPT2-L': [],
     'GPT3': [],
     'Llama2': [],
     'entropy': []}

# Get paths from config
run_name = config.training_config.get("run_name", "unnamed")
N_samples = config.inference.N_samples

# Load inference results from logs/{run_name}/inference/
inference_dir = f"logs/{run_name}/inference"
print(f"Loading inference results from: {inference_dir}")
for i in range(N_samples // B):
    texts = [torch.load(f"{inference_dir}/{i*B+j}.pt")["pred"][0] for j in range(B)]
    r = metric(texts)
    for key,value in r.items():
        res[key].append(value)
    print(r)
for key,value in res.items():
    res[key] = np.mean(res[key])

# Save results in logs/{run_name}/ directory
eval_output = f"logs/{run_name}/eval_unconditional.pt"
os.makedirs(os.path.dirname(eval_output), exist_ok=True)
torch.save(res, eval_output)
print(f"Evaluation results saved to: {eval_output}")


