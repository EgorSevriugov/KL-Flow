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
        
        # Update vocab_size from tokenizer if not specified
        if tokenizer is not None:
            if 'vocab_size' not in model_config or model_config['vocab_size'] is None:
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


def resize_model_embeddings(model, new_vocab_size, old_vocab_size=None):
    """
    Resize model token embeddings to accommodate new vocabulary size
    
    Args:
        model: Model instance
        new_vocab_size: New vocabulary size
        old_vocab_size: Old vocabulary size (optional, for validation)
    
    Returns:
        Updated model
    """
    if old_vocab_size is not None and new_vocab_size == old_vocab_size:
        return model
    
    print(f"Resizing model embeddings to {new_vocab_size}")
    
    # Try to detect model architecture and resize appropriately
    if hasattr(model, 'token_emb') and hasattr(model.token_emb, 'weight'):
        # For models using Linear embedding layer (new models)
        n_embd = model.n_embd
        
        # Resize input embedding
        old_weight = model.token_emb.weight.data
        model.token_emb = nn.Linear(new_vocab_size, n_embd, bias=False)
        model.token_emb.weight.data[:old_weight.size(0)] = old_weight
        
        # Resize output head
        old_weight = model.lm_head.weight.data
        model.lm_head = nn.Linear(n_embd, new_vocab_size, bias=False)
        model.lm_head.weight.data[:old_weight.size(0)] = old_weight
        
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        # For legacy models using transformer.wte
        n_embd = model.config.n_embd if hasattr(model, 'config') else model.n_embd
        
        # Resize input embedding
        old_weight = model.transformer.wte.weight.data
        model.transformer.wte = nn.Linear(new_vocab_size, n_embd, bias=False)
        model.transformer.wte.weight.data[:old_weight.size(0)] = old_weight
        
        # Resize output head
        old_weight = model.lm_head.weight.data
        model.lm_head = nn.Linear(n_embd, new_vocab_size, bias=False)
        model.lm_head.weight.data[:old_weight.size(0)] = old_weight
    else:
        print("Warning: Could not detect model architecture for embedding resize")
    
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
    ckpt = torch.load(checkpoint_path, map_location=device)
    
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
