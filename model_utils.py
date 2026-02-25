"""
Utility functions for loading and managing models
"""
import importlib
import torch
import torch.nn as nn


def load_class(module_name, class_name):
    """
    Dynamically load a class from a module
    
    Args:
        module_name: Module name (e.g., "FM_utils", "models.gpt_causal")
        class_name: Class name (e.g., "Logit", "GPTCausal")
    
    Returns:
        The loaded class
    """
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_model(config, fm_loss_func=None, tokenizer=None):
    """
    Load model from config with support for both new and legacy formats
    
    Args:
        config: OmegaConf config object
        fm_loss_func: Optional flow matching loss function
        tokenizer: Optional tokenizer for vocab size inference
    
    Returns:
        Loaded model instance
    """
    # Support both new config format (model.type) and legacy format (model_type)
    if hasattr(config.model, 'type') and config.model.type is not None:
        # New format: model.type specifies the class name directly
        model_class = load_class(f"models.{config.model_type}", config.model.type)
        
        # Create model config dict excluding 'type' field
        model_config = {k: v for k, v in config.model.items() if k != 'type'}
        
        # vocab_size is always taken from tokenizer when provided (no resizing later)
        if tokenizer is not None:
            model_config['vocab_size'] = len(tokenizer)
        
        # Create model with loss function if provided
        if fm_loss_func is not None:
            model = model_class(**model_config, loss_func=fm_loss_func)
        else:
            model = model_class(**model_config)
            
    else:
        # Legacy format not supported - old model files removed
        raise ValueError(
            f"Legacy model format detected. Please update your config to use the new format:\n"
            f"  model_type: 'flow_matching_transformer' or 'gpt_causal'\n"
            f"  model:\n"
            f"    type: 'FlowMatchingTransformer' or 'GPTCausal'\n"
            f"    vocab_size: {config.model.vocab_size}\n"
            f"    n_embd: {config.model.n_embd}\n"
            f"    n_layer: {config.model.n_layer}\n"
            f"    n_head: {config.model.n_head}\n"
            f"See NEW_MODELS_GUIDE.md for migration instructions."
        )
    
    return model


def load_checkpoint(model, checkpoint_path, device='cpu', strict=True):
    """
    Load model weights from checkpoint
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to map tensors to
        strict: Whether to strictly enforce state dict keys match
    
    Returns:
        Model with loaded weights
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    
    # Remove _orig_mod. prefix if present (from torch.compile)
    state_dict = {name.replace("_orig_mod.", ""): value 
                  for name, value in state_dict.items()}
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    
    return model
