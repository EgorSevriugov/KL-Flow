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

# from Eval_utils import EvalMetric

# metric = EvalMetric(device="cuda",max_length=config.data.sequence_length)

from tqdm import tqdm

import signal

# Define a timeout handler
def handler(signum, frame):
    raise TimeoutError("Function execution timed out")

# Function to set a timeout on your function
def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)  # Set the alarm for 'seconds'
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Disable the alarm
        return wrapper
    return decorator

@timeout(2)
def call_function(func):
    imports = "\n".join([line for line in func.split("\n") if line.startswith("import ")])
    loc = {}
    exec(imports,{},loc)
    res = exec(func,loc,{})
    return res

def pass_at_k(n, c, k):
    if n - c < k: 
        return 1.0 
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

# Get paths from config
run_name = config.training_config.get("run_name", "unnamed")

# Load inference results from logs/{run_name}/inference/
inference_dir = f"logs/{run_name}/inference"
print(f"Loading inference results from: {inference_dir}")
preds = []
test_lists = []
N_samples = min(config.inference.N_samples, len(os.listdir(inference_dir)))
res = []
res_compiled = []

success_ids = []
for i in range(N_samples):
    preds = torch.load(f"{inference_dir}/{i}.pt")["pred"]
    test_lists = torch.load(f"{inference_dir}/{i}.pt")["test_list"]
    c = 0
    N = len(preds)
    for j,pred in enumerate(preds):
        code = "\n".join([pred] + test_lists)
        try:
            _ = call_function(code)
            c += 1
            success_ids.append([i,j])
        except:
            pass
    code_tar = "\n".join([torch.load(f"{inference_dir}/{i}.pt")["tar"]] + test_lists)
    c_tar = 0
    try:
        _ = call_function(code_tar)
        c_tar += 1
    except:
        pass
    print(N,c)
    print(f"Pass@1: {pass_at_k(N,c,1)}, Pass@5: {pass_at_k(N,c,5)}, Pass@10: {pass_at_k(N,c,10)}, Tar: {c_tar}")
    res.append([c_tar]+[pass_at_k(N,c,k) for k in [1,5,10]])

for i in range(N_samples):
    preds = torch.load(f"{inference_dir}/{i}.pt")["pred"]
    test_lists = torch.load(f"{inference_dir}/{i}.pt")["test_list"]
    c = 0
    N = len(preds)
    for pred in preds:
        code = pred
        try:
            _ = call_function(code)
            c += 1
        except:
            pass
    code_tar = torch.load(f"{inference_dir}/{i}.pt")["tar"]
    c_tar = 0
    try:
        _ = call_function(code_tar)
        c_tar += 1
    except:
        pass
    print(f"Compiled@1: {pass_at_k(N,c,1)}, Compiled@5: {pass_at_k(N,c,5)}, Compiled@10: {pass_at_k(N,c,10)}, Tar: {c_tar}")
    res_compiled.append([pass_at_k(N,c,k) for k in [1,5,10]])

# Save results in logs/{run_name}/ directory
eval_output = f"logs/{run_name}/eval_infilling.pt"
success_ids_output = f"logs/{run_name}/eval_infilling_success_ids.pt"
os.makedirs(os.path.dirname(eval_output), exist_ok=True)
torch.save(np.concatenate([np.array(res).mean(axis=0), np.array(res_compiled).mean(axis=0)], axis=0), eval_output)
torch.save(success_ids, success_ids_output)
print(f"Evaluation results saved to: {eval_output}")
print(f"Success IDs saved to: {success_ids_output}")