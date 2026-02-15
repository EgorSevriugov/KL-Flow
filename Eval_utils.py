#!/usr/bin/env python
# coding: utf-8
# %%
import torch
from torchdiffeq import odeint
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchmetrics.text import Perplexity
import sys


# %%
def seq_entropy(seqs):
    entropy = 0
    for seq in seqs:
        _, counts = torch.unique(seq,return_counts=True)
        p = counts.float() / counts.sum()
        entropy += (-p * torch.log(p)).sum() / len(seqs)
    return entropy.item()


# %%
class EvalMetric(object):
    def __init__(self,device="cpu",max_length=2048):
        model_names = {
            "GPT2":"gpt2",
            "GPT2-L":"gpt2-large",
            "GPT3": "EleutherAI/gpt-neo-2.7B",
            "Llama2": "NousResearch/Llama-2-7b-chat-hf",
        }
        self.device = device
        self.max_length = max_length
        self.names = [name for name,_ in model_names.items()]
        self.tokenizers = [AutoTokenizer.from_pretrained(value) for _,value in model_names.items()]
        self.models = [AutoModelForCausalLM.from_pretrained(value) for _,value in model_names.items()]
        for i in range(len(self.models)):
            self.models[i].eval()
            self.models[i].to(device)
            self.tokenizers[i].pad_token = "[PAD]"
            self.tokenizers[i].pad_token_id = self.tokenizers[i].eos_token_id
            self.tokenizers[i].padding_side = "right"
    @torch.no_grad()
    def __call__(self,texts):
        res = {name:[] for name in self.names}
        res["entropy"] = []
        for name, tokenizer, model in zip(self.names,self.tokenizers,self.models):
            perp = Perplexity(ignore_index=tokenizer.eos_token_id).to(self.device)
            ids = []
            for text in texts:
                ids.append(tokenizer.encode(text,return_tensors="pt",padding="max_length",max_length=self.max_length)[:,:self.max_length])
            ids = torch.cat(ids).to(self.device)
            res["entropy"].append(seq_entropy(ids))
            log_probs = model(ids).logits.log_softmax(dim=-1)[:,:-1]
            res[name].append(perp(log_probs,ids[:,1:]).item())
        return {name:np.mean(value) for name, value in res.items()}
