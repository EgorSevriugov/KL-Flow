import os
import sys
from typing import Dict, Optional, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch._inductor.config as inductor_config

# FM utils and model utilities
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset
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
# Dataset Implementation (matching train_fm.py)
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


class InferenceCollator:
    """
    Data collator for inference.
    Tokenizes text, adds special tokens, and pads sequences.
    Does NOT apply flow matching - that happens in the inference loop.
    """
    def __init__(
        self, 
        tokenizer: AutoTokenizer,
        max_length: int,
        condition: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.condition = condition
        self.vocab_size = tokenizer.vocab_size
        
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Tokenize and pad sequences. Returns x1 (target) and mask for inference.
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
            
            # Pad to max length in batch
            max_len_in_batch = min(max(len(ids) for ids in all_input_ids), self.max_length)
            padded_input_ids = []
            masks = []
            
            eos_token_id = self.tokenizer.eos_token_id
            for prompt_ids, response_ids in zip(prompt_encoded["input_ids"], response_encoded["input_ids"]):
                # Structure: [BOS] prompt [EOS] response [EOS]
                combined_ids = prompt_ids + response_ids + [eos_token_id]
                # Create mask: 1 for prompt (includes BOS and EOS), 0 for response and final EOS (to be generated)
                mask = [1] * len(prompt_ids) + [0] * len(response_ids) + [0]
                
                # Pad
                pad_len = max_len_in_batch - len(combined_ids)
                if pad_len > 0:
                    combined_ids = combined_ids + [self.tokenizer.pad_token_id] * pad_len
                    mask = mask + [0] * pad_len
                
                padded_input_ids.append(combined_ids[:max_len_in_batch])
                masks.append(mask[:max_len_in_batch])
            
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
            
            # Pad to max length in batch
            max_len_in_batch = min(max(len(ids) for ids in encoded["input_ids"]), self.max_length)
            padded_input_ids = []
            masks = []
            
            for ids in encoded["input_ids"]:
                # Create mask: 1 for BOS token (first token), 0 for rest (to be generated)
                mask = [1] + [0] * (len(ids) - 1)
                
                # Pad
                pad_len = max_len_in_batch - len(ids)
                if pad_len > 0:
                    ids = ids + [self.tokenizer.pad_token_id] * pad_len
                    mask = mask + [0] * pad_len
                
                padded_input_ids.append(ids[:max_len_in_batch])
                masks.append(mask[:max_len_in_batch])
            
            input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
            condition_mask = torch.tensor(masks, dtype=torch.bool)
        
        # Return tokenized data for inference
        # Flow matching will be applied in the inference loop starting at t=0
        return {
            "x1": input_ids,
            "mask": condition_mask,
        }

# -----------------------------------------------------------------------------
# Main Inference Function
# -----------------------------------------------------------------------------

def find_checkpoint(config) -> str:
    """
    Find checkpoint path from config.
    
    Priority:
    1. config.inference.checkpoint (if specified)
    2. Auto-detect from logs/{run_name}/ directory (latest checkpoint)
    
    Args:
        config: OmegaConf configuration
    
    Returns:
        Path to checkpoint file
    
    Raises:
        ValueError: If no checkpoint can be found
    """
    # Check if explicitly specified
    if config.inference.checkpoint is not None and config.inference.checkpoint != "":
        ckpt_path = config.inference.checkpoint
        if os.path.exists(ckpt_path):
            print(f"Using checkpoint from config: {ckpt_path}")
            return ckpt_path
        else:
            print(f"Warning: Specified checkpoint not found: {ckpt_path}")
    
    # Try to auto-detect from training save path
    run_name = config.training_config.get("run_name", None)
    if run_name is not None:
        logs_dir = f"logs/{run_name}"
        
        if os.path.exists(logs_dir):
            # Look for checkpoint files
            import glob
            
            # Try different checkpoint patterns
            patterns = [
                os.path.join(logs_dir, "checkpoint-*", "*.pt"),  # HF Trainer format
                os.path.join(logs_dir, "ckpt_*.pt"),             # Custom format with step
                os.path.join(logs_dir, "ckpt.pt"),               # Final checkpoint
                os.path.join(logs_dir, "*.pt"),                  # Any .pt file
            ]
            
            for pattern in patterns:
                checkpoints = glob.glob(pattern)
                if checkpoints:
                    # Get the latest checkpoint (by modification time)
                    latest_ckpt = max(checkpoints, key=os.path.getmtime)
                    print(f"Auto-detected checkpoint: {latest_ckpt}")
                    return latest_ckpt
            
            print(f"Warning: No checkpoints found in {logs_dir}")
        else:
            print(f"Warning: Logs directory not found: {logs_dir}")
    
    # If we get here, no checkpoint found
    raise ValueError(
        "No checkpoint found. Please specify one of:\n"
        "  1. inference.checkpoint in config\n"
        "  2. Ensure training_config.run_name is set and logs/{run_name}/ exists with checkpoints"
    )


def main():
    # Determine checkpoint to load
    ckpt_path = find_checkpoint(config)
    print(f"Loading model from: {ckpt_path}")
    
    # Set up device
    assert torch.cuda.is_available(), "CUDA is required for inference"
    device = 'cuda:0'
    print(f"Using device: {device}")
    
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
    
    # Create validation dataset
    val_dataset = TextDataset(
        dataset_path=config.data.dataset_path,
        split="validation",
        text_field=config.data.get("text_field", "text"),
        prompt_field=config.data.get("prompt_field", None),
        response_field=config.data.get("response_field", None),
        condition=config.data.condition,
        max_samples=config.inference.N_samples,
    )
    
    # Create data collator (inference only - no flow matching in collator)
    data_collator = InferenceCollator(
        tokenizer=tokenizer,
        max_length=config.data.sequence_length,
        condition=config.data.condition,
    )
    
    # Create data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.inference.B_data,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
    )
    
    # Load model (vocab_size = len(tokenizer))
    model = load_model(config, tokenizer=tokenizer)
    
    # Load checkpoint
    model = load_checkpoint(model, ckpt_path, device="cpu", strict=False)
    print(f"Model loaded from checkpoint: {ckpt_path}")
    
    model = model.to(device)
    model.eval()
    
    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False
    
    # Set up inference
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    
    def model_forward(t, x):
        with ctx:
            return model(t, x, return_logits=True)[0]
    
    # Compile inference function
    inference_mixed = torch.compile(fm.inference_mixed, dynamic=False)
    
    # Create output directory in logs/{run_name}/inference/
    run_name = config.training_config.get("run_name", "unnamed")
    output_dir = f"logs/{run_name}/inference"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving inference results to: {output_dir}")
    
    # Run inference
    B_sub = config.inference.B_sub_data
    sample_idx = 0
    
    from tqdm import tqdm
    
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Running inference")):
        # Extract batch data
        x1 = batch["x1"].to(device)  # Target sequences (tokenized)
        mask = batch["mask"].to(device) if batch["mask"] is not None else None  # Condition mask
        
        # Sample initial noise state x0 from x1
        x0 = fm.sampler_0(x1)
        
        # Get x0 at t=0 (starting point for inference)
        t_start = torch.zeros(x0.size(0), device=device)
        batch_fm = fm.interpolate(x0, x1, mask=mask, t=t_start)
        if len(batch_fm) == 3:
            _, _, x0 = batch_fm
        else:
            _, x0 = batch_fm
        
        # Run inference from t=0 to t=max_t
        # Note: mask is always present now (even for unconditional with BOS token masked)
        mask_expanded = mask.repeat_interleave(B_sub, dim=0) if mask is not None else None
        x0_expanded = x0.repeat_interleave(B_sub, dim=0)
        x1_hat = inference_mixed(model_forward, x0_expanded, mask=mask_expanded.float() if mask_expanded is not None else None)
        
        # Decode results
        B = x1.size(0)
        for i in range(B):
            if config.data.condition and mask is not None:
                # Conditional: extract prompt and generated response
                cond_size = mask[i].int().sum(dim=-1).item()
                
                # Extract prompt
                prompt_ids = x1[i, :cond_size]
                prompt_text = tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=True)
                
                # Extract target response
                target_ids = x1[i, cond_size:]
                target_text = tokenizer.decode(target_ids.tolist(), skip_special_tokens=True)
                
                # Extract generated responses (multiple samples)
                predictions = []
                for j in range(B_sub):
                    pred_idx = i * B_sub + j
                    if len(x1_hat.shape) == 3:
                        pred_ids = x1_hat[pred_idx, cond_size:, :tokenizer.vocab_size].argmax(dim=-1)
                    else:
                        pred_ids = x1_hat[pred_idx, cond_size:]
                    pred_text = tokenizer.decode(pred_ids.tolist(), skip_special_tokens=True)
                    if len(pred_text) > 0:
                        predictions.append(pred_text)
                
                # Save results
                if len(predictions) > 0:
                    torch.save(
                        {
                            "pred": predictions,
                            "prompt": prompt_text,
                            "target": target_text,
                        },
                        f"{output_dir}/{sample_idx}.pt"
                    )
                    sample_idx += 1
            else:
                # Unconditional: just decode the generated text
                predictions = []
                for j in range(B_sub):
                    pred_idx = i * B_sub + j
                    if len(x1_hat.shape) == 3:
                        pred_ids = x1_hat[pred_idx, :, :tokenizer.vocab_size].argmax(dim=-1)
                    else:
                        pred_ids = x1_hat[pred_idx]
                    pred_text = tokenizer.decode(pred_ids.tolist(), skip_special_tokens=True)
                    if len(pred_text) > 0:
                        predictions.append(pred_text)
                
                # Save results
                if len(predictions) > 0:
                    torch.save(
                        {
                            "pred": predictions,
                            "prompt": None,
                            "target": None,
                        },
                        f"{output_dir}/{sample_idx}.pt"
                    )
                    sample_idx += 1
    
    print(f"\nInference completed! Generated {sample_idx} samples.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


