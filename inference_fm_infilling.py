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
import torch.distributed as dist
import torch._inductor.config as inductor_config
from optimizer import Muon
import wandb

# FM utils and model utilities
from omegaconf import OmegaConf
from model_utils import load_class, load_model, load_checkpoint


# -----------------------------------------------------------------------------
# Config Loading and Merging
# -----------------------------------------------------------------------------

def load_config(experiment_config_path: str) -> OmegaConf:
    """
    Load and merge default config with experiment config.
    
    Args:
        experiment_config_path: Path to experiment-specific config file
    
    Returns:
        Merged configuration
    """
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, "configs", "default_config.yaml")
    
    # Load default config
    if os.path.exists(default_config_path):
        default_config = OmegaConf.load(default_config_path)
        print(f"Loaded default config from: {default_config_path}")
    else:
        print(f"Warning: Default config not found at {default_config_path}")
        default_config = OmegaConf.create()
    
    # Load experiment config
    experiment_config = OmegaConf.load(experiment_config_path)
    print(f"Loaded experiment config from: {experiment_config_path}")
    
    # Merge configs (experiment config overrides defaults)
    config = OmegaConf.merge(default_config, experiment_config)
    print("Merged default and experiment configs")
    
    return config


# Load and merge configs
config = load_config(sys.argv[1])

# FM will be initialized after tokenizer is loaded


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

import boto3
from botocore.config import Config as boto_config

config_boto = boto_config(
    read_timeout=2000,
    connect_timeout=2000,
    retries={"max_attempts": 13}
)
session = boto3.session.Session(profile_name='default')
s3 = session.client(
   service_name='s3',
   endpoint_url='https://s3.cloud.ru',
    config=config_boto,
)


def _peek_data_shard(filename,iters_try=1000):
    # only reads the header, returns header data
    try:
        with open(filename, "rb") as f:
            # first read the header, which is 256 int32 integers (4 bytes each)
            header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520
        assert header[1] == 1
        ntok = header[2] # number of tokens (claimed)
        return ntok # for 
    except:
        pass
    for i in range(iters_try):
        try:
            f = s3.get_object(Bucket='fineweb',Key=filename.split("/")[-1])['Body']
            header = np.frombuffer(f.read(256*4), dtype=np.int32)
            assert header[0] == 20240520
            assert header[1] == 1
            ntok = header[2] # number of tokens (claimed)
            return ntok # for now just return the number of tokens
        except:
            pass
    assert False, "Iters to read file is out"

def _load_data_shard(filename,iters_try=1000):
    try:
        with open(filename, "rb") as f:
            # first read the header, which is 256 int32 integers (4 bytes each)
            header = np.frombuffer(f.read(256*4), dtype=np.int32)
            assert header[0] == 20240520, "magic number mismatch in the data .bin file"
            assert header[1] == 1, "unsupported version"
            ntok = header[2] # number of tokens (claimed)
            # the rest of it are tokens, stored as uint16
            if num_vocab < 2**16:
                dtype = np.uint16
            else:
                dtype = np.uint32
            tokens = np.frombuffer(f.read(), dtype=dtype)
        assert len(tokens) == ntok, "number of tokens read does not match header?"
        return tokens
    except:
        pass
    for i in range(iters_try):
        try:
            f = s3.get_object(Bucket='fineweb',Key=filename.split("/")[-1])['Body']
            # first read the header, which is 256 int32 integers (4 bytes each)
            header = np.frombuffer(f.read(256*4), dtype=np.int32)
            assert header[0] == 20240520, "magic number mismatch in the data .bin file"
            assert header[1] == 1, "unsupported version"
            ntok = header[2] # number of tokens (claimed)
            # the rest of it are tokens, stored as uint16
            if num_vocab < 2**16:
                dtype = np.uint16
            else:
                dtype = np.uint32
            tokens = np.frombuffer(f.read(), dtype=dtype)
            assert len(tokens) == ntok, "number of tokens read does not match header?"
            return tokens
        except:
            pass
    assert False, "Iters to read file is out"

class DistributedDataLoader:
    def __init__(self, vocab_size, filename_pattern, B, T, process_rank, num_processes,num_tokens_done=0, condition=False):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        if condition:
            self.T *= 2
            self.T = self.T + 1
        self.vocab_size = vocab_size
        # filename pattern up to last /
        self.path = "/".join(filename_pattern.split("/")[:-1])
        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total
        self.num_tokens_done = num_tokens_done
        self.condition = condition
        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0 + self.num_tokens_done // (10**8)
        self.current_position = self.process_rank * self.B * self.T + self.num_tokens_done % (10**8)
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x1 = buf.view(B, T) # targets
        idx, x1 = x1[:,0], x1[:,1:]
        mask = None
        if self.condition:
            x1, mask = x1.chunk(2,dim=1)
            mask_cond = (mask > 0.5).cuda()
            mask_ques = (mask == 2).cuda()
        x1 = x1.cuda()
        x0 = fm.sampler_0(x1)
        t,xt = fm.interpolate(x0,x1,mask=mask_cond)
        test_list = torch.load(os.path.join(self.path, f"{idx[0].item()}.pt"))
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes+1) > len(self.tokens):
            self.advance()
        return xt, x1, mask_cond, mask_ques, test_list


# -

# -----------------------------------------------------------------------------
# int main

class Hyperparameters:
    input_bin = config.data.input_bin
    input_val_bin = config.data.input_val_bin
    data_condition = config.data.condition
    input_bin_pretrain = config.data.input_bin_pretrain
    input_val_bin_pretrain = config.data.input_val_bin_pretrain
    num_tokens_to_train = config.training_config.num_tokens_to_train * 10**9
    batch_size = config.training_config.batch_size 
    device_batch_size = config.training_config.device_batch_size
    sequence_length = config.data.sequence_length
    embed_learning_rate = config.optimizer.embed_learning_rate
    muon_learning_rate = config.optimizer.muon_learning_rate
    warmup_iters = config.optimizer.warmup_iters
    warmdown_iters = config.optimizer.warmdown_iters 
    weight_decay = config.optimizer.weight_decay
    val_loss_every = config.training_config.val_loss_every
    save_every = config.training_config.save_every
    project_name = config.training_config.project_name
    run_name = config.training_config.run_name
    checkpoint = config.inference.checkpoint
    pretrain = config.training_config.pretrain
args = Hyperparameters()
postfix = sys.argv[2]
if args.checkpoint is not None:
    ckpt_path = args.checkpoint
else:
    assert False, "Define path to model checkpoint"
# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
ddp_rank = 0
ddp_world_size = 1
device = f'cuda:{ddp_rank}'
print(f"using device: {device}")
#master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.
master_process = True
# convenience variables
num_vocab = config.model.vocab_size
B, T = config.inference.B_data, args.sequence_length
B_sub = config.inference.B_sub_data


# train_loader = DistributedDataLoader(num_vocab, args.input_bin, B, T, ddp_rank, ddp_world_size,condition=args.data_condition)
val_loader = DistributedDataLoader(num_vocab, args.input_val_bin.replace("postfix",postfix), B, T, ddp_rank, ddp_world_size,condition=args.data_condition)

# +
import tiktoken
class CustomTokenizer(object):
    def __init__(self,name,special_tokens={}):
        self.name = name
        self.tokenizer = tiktoken.get_encoding(name)
        self.ind_to_str = {i:self.tokenizer.decode([i]) for i in range(self.tokenizer.n_vocab)}
        for key,value in special_tokens.items():
            self.ind_to_str[key] = value
        self.n_vocab = self.tokenizer.n_vocab + len(special_tokens)
        self.vocab_size = self.n_vocab
        # Add mask_token_id for compatibility
        self.mask_token_id = self.n_vocab - 1
    def encode(self,text):
        return self.tokenizer.encode(text)
    def decode(self,ind):
        return "".join([self.ind_to_str[idx] for idx in ind])
    def __len__(self):
        return self.n_vocab

tokenizer = CustomTokenizer("gpt2",special_tokens={50257:"<|pad|>"})

# Initialize FM utils with tokenizer
fm_type = load_class("FM_utils", config.fm.type)
fm = fm_type(config.fm_config, tokenizer=tokenizer)
print(f"FM initialized with vocab_size: {fm.vocab_size}")

# +
# Load model using utility function
model = load_model(config, fm_loss_func=fm.loss)

# Load checkpoint
model = load_checkpoint(model, ckpt_path, device="cpu", strict=True)
if master_process:
    print("Model loaded from checkpoint")
    
model = model.cuda()
if hasattr(inductor_config, "coordinate_descent_tuning"):
    inductor_config.coordinate_descent_tuning = True # suggested by @Chillee
# model = torch.compile(model)
for param in model.parameters():
    param.requires_grad = False
# here we wrap model into DDP container
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

model.eval()
def model_forward(t,x):
    with ctx:
        return model(t,x,return_logits=True)[0]
from tqdm import tqdm


inference_mixed = torch.compile(fm.inference_mixed,dynamic=False)
# inference_mixed = fm.inference_mixed
N_samples = min(config.inference.N_samples,int(val_loader.ntok_total / (2*T + 1)))
print(N_samples)

# Create output directory in logs/{run_name}/inference/
output_dir = f"logs/{run_name}/inference"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving inference results to: {output_dir}")
def process_code(code):
    # number of <|endoftext|> in the code equals to the twice the number of masked lines in the code plus 1
    parts = code.split("<|endoftext|>")
    code_processed = parts[0]
    parts = parts[1:]
    # print(parts)
    # assert len(parts) % 2 == 0, "number of <|endoftext|> in the code is not equal to the twice the number of masked lines in the code plus 1"
    n_masked = len(parts) // 2
    for i in range(n_masked):
        code_processed += parts[-n_masked+i] + parts[i]
    return code_processed.replace("<|pad|>","").replace("<|endoftext|>","")

n_vocab = tokenizer.n_vocab
for i in range(N_samples // B):
    x0, x1, mask_cond, mask_ques, test_list = val_loader.next_batch()
    mask_cond, x1 = mask_cond.repeat(B_sub,1).bool(), x1.repeat(B_sub,1)
    assert B == 1, "conditional inference not implemented for B > 1"
    cond_size = mask_ques[0].int().sum(dim=-1).item()
    fm.update_N(torch.logical_not(mask_cond[0]).int().sum(dim=-1).item())
    x0 = fm.sampler_0(x1)
    _,x0 = fm.interpolate(x0,x1,mask=mask_cond,t=torch.zeros(x0.size(0),device=device))
    
    x1_hat = inference_mixed(model_forward,x0,mask=mask_cond.float())

    if config.fm.type == "GPT":
        process = process_code
    else:
        process = lambda x: x.replace("<|pad|>","").replace("<|endoftext|>","")
    if config.fm.type == "GPT":
        pred = [process(tokenizer.decode(x1_hat[i,cond_size:].tolist())) for i in range(x1_hat.size(0))]
    else:
        pred = [process(tokenizer.decode(x1_hat[i,cond_size:,:n_vocab].argmax(dim=-1).tolist())) for i in range(x1_hat.size(0))]
    tar = [process(tokenizer.decode(x1[0,cond_size:].tolist()))]
    
    ques = [tokenizer.decode(x1[0,:cond_size].tolist())]
    pred = [p for p in pred if len(p) > 0]
    if len(pred) == 0:
        continue
    for j in range(B):
        torch.save({"pred":pred[j*B_sub:(j+1)*B_sub],
                    "ques":ques[j],
                    "tar":tar[j],
                    "test_list":test_list,
                   },
                   f"{output_dir}/{i*B+j}.pt"
                  )


