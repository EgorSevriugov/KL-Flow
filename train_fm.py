import os
import sys
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments, TrainerCallback, AutoTokenizer
from datasets import load_from_disk, load_dataset

# Import Muon optimizer from official package
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

# Import model utilities
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
# Dataset Implementation
# -----------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Dataset for loading raw text from HuggingFace datasets.
    Tokenization happens in the collator during batch formation.
    """
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        text_field: str = "text",
        prompt_field: Optional[str] = None,
        response_field: Optional[str] = None,
        condition: bool = False,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.text_field = text_field
        self.prompt_field = prompt_field
        self.response_field = response_field
        self.condition = condition
        
        # Load dataset
        try:
            dataset = load_from_disk(dataset_path)
            if hasattr(dataset, split):
                self.dataset = dataset[split]
            else:
                self.dataset = dataset
        except:
            self.dataset = load_dataset(dataset_path, split=split)
        
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a single example"""
        example = self.dataset[idx]
        
        if self.condition and self.prompt_field and self.response_field:
            # Conditional training with prompt/response
            prompt = example.get(self.prompt_field, "")
            response = example.get(self.response_field, "")
            return {
                "prompt": prompt,
                "response": response,
                "is_conditional": True
            }
        else:
            # Unconditional training
            text = example.get(self.text_field, "")
            return {
                "text": text,
                "is_conditional": False
            }


class FlowMatchingCollator:
    """
    Data collator for flow matching training.
    Tokenizes text, adds special tokens, pads sequences, and applies flow matching.
    """
    def __init__(
        self, 
        fm, 
        tokenizer: AutoTokenizer,
        max_length: int,
        condition: bool = False,
    ):
        self.fm = fm
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.condition = condition
        self.vocab_size = tokenizer.vocab_size
        
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Tokenize, pad, and apply flow matching interpolation to batch.
        """
        # Tokenize based on conditional or unconditional mode
        if self.condition and features[0].get("is_conditional", False):
            # Conditional: tokenize prompt and response separately
            prompts = [f["prompt"] for f in features]
            responses = [f["response"] for f in features]
            
            # Tokenize prompts (with BOS/EOS)
            prompt_encoded = self.tokenizer(
                prompts,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length // 2,
                padding=False,
                return_tensors=None,
            )
            
            # Tokenize responses (without special tokens - it's a continuation)
            response_encoded = self.tokenizer(
                responses,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length // 2,
                padding=False,
                return_tensors=None,
            )
            
            # Combine prompt and response for each example
            # Structure: [BOS] prompt [EOS] response [EOS]
            all_input_ids = []
            eos_token_id = self.tokenizer.eos_token_id
            for prompt_ids, response_ids in zip(prompt_encoded["input_ids"], response_encoded["input_ids"]):
                # Prompt already has BOS and EOS from add_special_tokens=True
                # Response has no special tokens, so we add EOS at the end
                combined_ids = prompt_ids + response_ids + [eos_token_id]
                all_input_ids.append(combined_ids)
            
            # Pad to config max sequence length (fixed length per batch)
            pad_length = self.max_length
            padded_input_ids = []
            masks = []
            
            eos_token_id = self.tokenizer.eos_token_id
            for prompt_ids, response_ids in zip(prompt_encoded["input_ids"], response_encoded["input_ids"]):
                # Structure: [BOS] prompt [EOS] response [EOS]
                combined_ids = prompt_ids + response_ids + [eos_token_id]
                # Create mask: 1 for prompt (includes BOS and EOS), 0 for response and final EOS (to be generated)
                mask = [1] * len(prompt_ids) + [0] * len(response_ids) + [0]
                
                pad_len = pad_length - len(combined_ids)
                if pad_len > 0:
                    combined_ids = combined_ids + [self.tokenizer.pad_token_id] * pad_len
                    mask = mask + [0] * pad_len
                
                padded_input_ids.append(combined_ids[:pad_length])
                masks.append(mask[:pad_length])
            
            input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
            condition_mask = torch.tensor(masks, dtype=torch.bool)
            
        else:
            # Unconditional: tokenize text
            texts = [f["text"] for f in features]
            
            # Tokenize with special tokens
            encoded = self.tokenizer(
                texts,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )
            
            # Pad to config max sequence length (fixed length per batch)
            pad_length = self.max_length
            padded_input_ids = []
            masks = []
            
            for ids in encoded["input_ids"]:
                # Create mask: 1 for BOS token (first token), 0 for rest (to be generated)
                mask = [1] + [0] * (len(ids) - 1)
                
                pad_len = pad_length - len(ids)
                if pad_len > 0:
                    ids = ids + [self.tokenizer.pad_token_id] * pad_len
                    mask = mask + [0] * pad_len
                
                padded_input_ids.append(ids[:pad_length])
                masks.append(mask[:pad_length])
            
            input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
            condition_mask = torch.tensor(masks, dtype=torch.bool)
        
        # Apply flow matching
        x1 = input_ids
        x0 = self.fm.sampler_0(x1)
        t, xt = self.fm.interpolate(x0, x1, mask=condition_mask)
        
        return {
            "t": t,
            "xt": xt,
            "labels": x1,
        }


# -----------------------------------------------------------------------------
# Trainer with Flow Matching Loss
# -----------------------------------------------------------------------------

class FlowMatchingTrainer(Trainer):
    """
    Standard HuggingFace Trainer with custom loss computation for flow matching.
    """

    def get_train_dataloader(self):
        """Build train dataloader with prefetch_factor from config (for precomputing batches)."""
        prefetch = config.training_config.get("dataloader_prefetch_factor", 4)
        if getattr(self.args, "dataloader_num_workers", 0) > 0:
            # Ensure prefetch_factor is set (works even if TrainingArguments didn't accept it at init)
            setattr(self.args, "dataloader_prefetch_factor", prefetch)
        return super().get_train_dataloader()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Compute flow matching loss.
        """
        t = inputs.get("t")
        xt = inputs.get("xt")
        labels = inputs.get("labels")
        
        # Forward pass
        _, loss = model(t, xt, labels, return_logits=False)
        
        return (loss, {"loss": loss}) if return_outputs else loss
    
    def create_optimizer(self):
        """
        Create MuonWithAuxAdam optimizer with proper parameter grouping.
        """
        if self.optimizer is None:
            weight_decay = config.optimizer.weight_decay
            ref_batch = config.training_config.get("reference_batch_size", 128)
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            effective_batch_size = config.training_config.device_batch_size * world_size
            lr_scale = effective_batch_size / ref_batch
            muon_lr = config.optimizer.muon_learning_rate * lr_scale
            embed_lr = config.optimizer.embed_learning_rate * lr_scale

            raw_model = self.model
            if hasattr(raw_model, 'module'):
                raw_model = raw_model.module
            if hasattr(raw_model, '_orig_mod'):
                raw_model = raw_model._orig_mod

            all_params = list(raw_model.parameters())
            two_d = [p for p in all_params if p.ndim >= 2]
            other = [p for p in all_params if p.ndim < 2]

            param_groups = [
                dict(params=two_d, use_muon=True, lr=muon_lr, momentum=0.95, weight_decay=weight_decay),
                dict(params=other, use_muon=False, lr=embed_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=weight_decay),
            ]
            
            # Choose optimizer based on distributed setup
            if dist.is_initialized():
                optimizer_cls = MuonWithAuxAdam
            else:
                optimizer_cls = SingleDeviceMuonWithAuxAdam
            
            self.optimizer = optimizer_cls(param_groups)
        
        return self.optimizer
    
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        Create learning rate scheduler with linear warmup and warmdown.
        """
        warmup_iters = config.optimizer.warmup_iters
        warmdown_iters = config.optimizer.warmdown_iters
        
        def get_lr(current_step):
            """Linear warmup and warmdown"""
            if current_step < warmup_iters:
                return (current_step + 1) / warmup_iters
            elif current_step < num_training_steps - warmdown_iters:
                return 1.0
            else:
                decay_ratio = (num_training_steps - current_step) / warmdown_iters
                return decay_ratio
        
        if optimizer is None:
            optimizer = self.optimizer
        
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
        
        return self.lr_scheduler


# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------

def main():
    # Determine checkpoint to load
    ckpt_path = None
    resume_from_checkpoint = None
    
    if config.training_config.checkpoint is not None:
        resume_from_checkpoint = config.training_config.checkpoint
        ckpt_path = config.training_config.checkpoint
    elif config.training_config.pretrain is not None:
        ckpt_path = config.training_config.pretrain
    
    # Effective batch size (no gradient accumulation): per-device batch × num devices
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    effective_batch_size = config.training_config.device_batch_size * world_size

    # Calculate training steps from target token count
    num_tokens = config.training_config.num_tokens_to_train * 10**9
    tokens_per_step = effective_batch_size * config.data.sequence_length
    num_training_steps = int(num_tokens / tokens_per_step)
    
    # Load tokenizer
    print(f"Loading tokenizer: {config.data.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    
    # Add special tokens if not present
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "<|PAD|>"
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = "<|BOS|>"
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = "<|EOS|>"
    if tokenizer.mask_token is None:
        special_tokens_dict["mask_token"] = "<|MASK|>"
    
    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added special tokens: {special_tokens_dict}")
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Initialize FM utils with tokenizer
    fm_type = load_class("FM_utils", config.fm.type)
    fm = fm_type(config.fm_config, tokenizer=tokenizer)
    print(f"FM initialized with vocab_size: {fm.vocab_size}")
    
    # Create datasets
    train_dataset = TextDataset(
        dataset_path=config.data.dataset_path,
        split="train",
        text_field=config.data.get("text_field", "text"),
        prompt_field=config.data.get("prompt_field", None),
        response_field=config.data.get("response_field", None),
        condition=config.data.condition,
    )
    
    eval_dataset = TextDataset(
        dataset_path=config.data.dataset_path,
        split="validation",
        text_field=config.data.get("text_field", "text"),
        prompt_field=config.data.get("prompt_field", None),
        response_field=config.data.get("response_field", None),
        condition=config.data.condition,
    )
    
    # Create data collator
    data_collator = FlowMatchingCollator(
        fm=fm,
        tokenizer=tokenizer,
        max_length=config.data.sequence_length,
        condition=config.data.condition,
    )
    
    # Create model (vocab_size = len(tokenizer))
    model = load_model(config, fm_loss_func=None, tokenizer=tokenizer)
    
    # Load checkpoint if specified
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        model = load_checkpoint(model, ckpt_path, device="cpu", strict=False)
        print(f"Model loaded from checkpoint: {ckpt_path}")
    
    # Compile model (requires Triton; if gcc fails, set CUDA_HOME and ensure gcc finds CUDA headers)
    if hasattr(torch, '_inductor') and hasattr(torch._inductor, 'config'):
        torch._inductor.config.coordinate_descent_tuning = True
    # try:
    #     # dynamic=True: allow variable sequence length per batch without recompilation
    #     model = torch.compile(model, dynamic=True)
    #     print("Model compiled with torch.compile (dynamic=True for variable sequence length)")
    # except Exception as e:
    #     err = str(e)
    #     if "gcc" in err.lower() or "triton" in err.lower() or "CalledProcessError" in err:
    #         print(
    #             "\nTriton compilation failed (gcc/CUDA). To fix:\n"
    #             "  1. Set CUDA_HOME to your CUDA install, e.g.: export CUDA_HOME=/usr/local/cuda\n"
    #             "  2. Ensure gcc can find CUDA headers: install system CUDA toolkit (e.g. cuda-nvcc-12-6)\n"
    #             "  3. Or run in a conda env with: conda install cuda-nvcc -c nvidia\n"
    #             "  4. Verify: echo $CUDA_HOME && ls $CUDA_HOME/include/cuda.h\n"
    #         )
    #     raise

    # No gradient accumulation: LR and steps are tuned for effective_batch_size
    gradient_accumulation_steps = 1
    ref_batch = config.training_config.get("reference_batch_size", 128)
    lr_scale = effective_batch_size / ref_batch
    if lr_scale != 1.0:
        print(f"Effective batch size {effective_batch_size} (ref={ref_batch}): scaling LR by {lr_scale:.4f}")

    # Training arguments
    output_dir = f'logs/{config.training_config.run_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=num_training_steps,
        per_device_train_batch_size=config.training_config.device_batch_size,
        per_device_eval_batch_size=config.training_config.device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="steps" if config.training_config.val_loss_every > 0 else "no",
        eval_steps=config.training_config.val_loss_every if config.training_config.val_loss_every > 0 else None,
        save_strategy="steps",
        save_steps=config.training_config.save_every,
        save_total_limit=1,  # only last checkpoint (overwritten every save_every)
        logging_steps=1,
        logging_first_step=True,
        logging_dir=f'{output_dir}/logs',
        bf16=True,
        dataloader_num_workers=32,
        dataloader_prefetch_factor=config.training_config.get("dataloader_prefetch_factor", 4),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        local_rank=int(os.environ.get('LOCAL_RANK', -1)),
        report_to=["tensorboard"],
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
    )
    
    # Create trainer
    trainer = FlowMatchingTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    trainer.save_model(output_dir)
    
    print(f"Training completed. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
