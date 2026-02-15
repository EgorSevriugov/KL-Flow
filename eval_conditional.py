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
from evaluate import load as load_metric


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

# Get paths from config
run_name = config.training_config.get("run_name", "unnamed")
N_samples = config.inference.N_samples

# Load inference results from logs/{run_name}/inference/
inference_dir = f"logs/{run_name}/inference"
print(f"Loading inference results from: {inference_dir}")
preds = []
tars = []
for i in range(N_samples):
    preds.append(torch.load(f"{inference_dir}/{i}.pt")["pred"])
    tars.append(torch.load(f"{inference_dir}/{i}.pt")["tar"])
def eval_func(predictions,references):
    bleu = load_metric("bleu")
    rouge = load_metric('rouge')
    bertscore = load_metric("bertscore", module_type="metric")
    def eval_text(predictions,references):
        bleu_score = bleu.compute(references=[references], predictions=[predictions], max_order=4, smooth=False)["bleu"]
        rouge_score = rouge.compute(predictions=[predictions], references=[references])["rougeL"]
        bert_score = np.mean(bertscore.compute(predictions=[predictions], references=[references], model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)["f1"])
        return np.array([bleu_score,rouge_score,bert_score])
    top_scores = []
    mean_scores = []
    for pred, tar in zip(preds,tars):
        try:
            score = []
            for p in pred:
                score.append(eval_text(p,tar))
            score = np.stack(score)
            top_scores.append(score.max(axis=0))
            mean_scores.append(score.mean(axis=0))
            print(top_scores[-1],mean_scores[-1])
        except:
            pass
    return np.stack(top_scores).mean(axis=0), np.stack(mean_scores).mean(axis=0)
fres = eval_func(preds,tars)
res = {
    "top5":{"bleu":fres[0][0],"rouge":fres[0][1],"bert":fres[0][2]},
    "mean":{"bleu":fres[1][0],"rouge":fres[1][1],"bert":fres[1][2]}
}

# Save results in logs/{run_name}/ directory
eval_output = f"logs/{run_name}/eval_conditional.pt"
os.makedirs(os.path.dirname(eval_output), exist_ok=True)
torch.save(res, eval_output)
print(f"Evaluation results saved to: {eval_output}")


