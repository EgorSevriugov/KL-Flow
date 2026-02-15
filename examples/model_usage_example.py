"""
Example script demonstrating usage of both model architectures:
1. GPTCausal - for autoregressive language modeling
2. FlowMatchingTransformer - for flow matching training
"""

import torch
import torch.nn.functional as F
from models.gpt_causal import GPTCausal
from models.flow_matching_transformer import FlowMatchingTransformer


def example_gpt_causal():
    """Example of using GPTCausal for autoregressive generation"""
    print("=" * 50)
    print("GPT Causal Model Example")
    print("=" * 50)
    
    # Initialize model
    model = GPTCausal(
        vocab_size=50304,
        n_embd=768,
        n_layer=12,
        n_head=12,
        dropout=0.1,
        max_seq_len=2048
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Example 1: Training forward pass
    print("\n1. Training forward pass:")
    batch_size = 4
    seq_len = 128
    
    # Input: token indices
    tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(tokens, targets=targets)
    print(f"   Input shape: {tokens.shape}")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Loss: {loss.item():.4f}")
    
    # Example 2: Autoregressive generation
    print("\n2. Autoregressive generation:")
    model.eval()
    with torch.no_grad():
        # Start with some prompt tokens
        prompt = torch.randint(0, model.vocab_size, (1, 10))
        print(f"   Prompt shape: {prompt.shape}")
        
        # Generate 20 new tokens
        generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
        print(f"   Generated shape: {generated.shape}")
        print(f"   Generated {generated.shape[1] - prompt.shape[1]} new tokens")


def example_flow_matching():
    """Example of using FlowMatchingTransformer for flow matching"""
    print("\n" + "=" * 50)
    print("Flow Matching Transformer Example")
    print("=" * 50)
    
    # Initialize model
    model = FlowMatchingTransformer(
        vocab_size=50304,
        n_embd=768,
        n_layer=12,
        n_head=12,
        dropout=0.1,
        max_seq_len=2048,
        time_embed_dim=256
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Example 1: Training forward pass
    print("\n1. Training forward pass:")
    batch_size = 4
    seq_len = 128
    
    # Input: timesteps (scalar per sample)
    t = torch.rand(batch_size)
    
    # Input: one-hot or soft token distributions
    xt = F.one_hot(torch.randint(0, model.vocab_size, (batch_size, seq_len)), 
                   num_classes=model.vocab_size).float()
    
    # Targets: can be token indices or distributions
    targets = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(t, xt, targets=targets)
    print(f"   Timestep shape: {t.shape}")
    print(f"   Input shape: {xt.shape}")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Loss: {loss.item():.4f}")
    
    # Example 2: Sampling at specific timestep
    print("\n2. Sampling at timestep:")
    model.eval()
    with torch.no_grad():
        t_sample = torch.tensor([0.5, 0.5, 0.5, 0.5])  # Timestep for each sample
        
        # Current state (e.g., noised tokens)
        xt_sample = F.one_hot(torch.randint(0, model.vocab_size, (batch_size, seq_len)),
                             num_classes=model.vocab_size).float()
        
        # Get model predictions
        logits = model.sample(t_sample, xt_sample, temperature=1.0)
        probs = F.softmax(logits, dim=-1)
        
        print(f"   Timestep: {t_sample[0].item():.2f}")
        print(f"   Output probabilities shape: {probs.shape}")
        print(f"   Can use these for flow matching iterative refinement")


def example_flow_matching_with_fm_utils():
    """Example integrating FlowMatchingTransformer with FM_utils"""
    print("\n" + "=" * 50)
    print("Flow Matching with FM Utils Integration")
    print("=" * 50)
    
    # Import FM utils
    from FM_utils import Logit
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    
    # Load tokenizer (required for FM initialization)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = len(tokenizer)
    
    # Initialize FM config
    fm_config = OmegaConf.create({
        "max_t": 0.3,
        "N": 1024,
        "beta": 0.01,
        "t_split": 0.28
    })
    
    # Initialize FM sampler with tokenizer
    fm = Logit(fm_config, tokenizer=tokenizer)
    print(f"FM type: {type(fm).__name__}")
    print(f"FM vocab_size: {fm.vocab_size}")
    
    # Initialize model with FM loss
    model = FlowMatchingTransformer(
        vocab_size=fm.vocab_size,
        n_embd=768,
        n_layer=6,  # Smaller for example
        n_head=6,
        dropout=0.1,
        max_seq_len=512,
        loss_func=fm.loss  # Use FM loss function
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Example training step
    print("\n1. Flow matching training step:")
    batch_size = 4
    seq_len = 128
    
    # Sample target token indices
    x1_indices = torch.randint(0, fm.vocab_size, (batch_size, seq_len))
    
    # Sample timesteps
    t = torch.rand(batch_size)
    
    # Sample initial state (uniform noise) - now sampler_0 takes token indices
    x0 = fm.sampler_0(x1_indices)
    
    # Interpolate to get xt
    _, xt = fm.interpolate(x0, x1_indices, t)
    
    # Convert x1_indices to one-hot for loss computation
    x1 = F.one_hot(x1_indices, num_classes=fm.vocab_size).float()
    
    # Forward pass
    logits, loss = model(t, xt, targets=x1)
    
    print(f"   Timesteps: {t.cpu().numpy()}")
    print(f"   xt shape: {xt.shape}")
    print(f"   x1 shape: {x1.shape}")
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n   This loss can be used for gradient descent!")


def compare_models():
    """Compare the two model architectures"""
    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)
    
    config = {
        "vocab_size": 50304,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "dropout": 0.1,
        "max_seq_len": 2048
    }
    
    gpt = GPTCausal(**config)
    fm = FlowMatchingTransformer(**config, time_embed_dim=256)
    
    gpt_params = sum(p.numel() for p in gpt.parameters()) / 1e6
    fm_params = sum(p.numel() for p in fm.parameters()) / 1e6
    
    print(f"\nGPT Causal:")
    print(f"  - Parameters: {gpt_params:.2f}M")
    print(f"  - Attention: Causal (autoregressive)")
    print(f"  - Position encoding: Rotary embeddings")
    print(f"  - Time conditioning: None")
    print(f"  - Use case: Standard language modeling")
    
    print(f"\nFlow Matching Transformer:")
    print(f"  - Parameters: {fm_params:.2f}M")
    print(f"  - Attention: Bidirectional (full)")
    print(f"  - Position encoding: None (uses time)")
    print(f"  - Time conditioning: Yes (timestep embedding)")
    print(f"  - Use case: Flow matching, discrete diffusion")
    
    print(f"\nParameter difference: {abs(gpt_params - fm_params):.2f}M")


if __name__ == "__main__":
    # Run all examples
    example_gpt_causal()
    example_flow_matching()
    example_flow_matching_with_fm_utils()
    compare_models()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("=" * 50)
