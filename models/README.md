# Model Architecture Documentation

This directory contains transformer-based models for different training paradigms.

## Models

### 1. `gpt_causal.py` - GPT-style Causal Transformer

**Purpose**: Autoregressive language modeling with causal (left-to-right) attention.

**Key Features**:
- **Causal Attention**: Each token can only attend to previous tokens
- **Rotary Position Embeddings (RoPE)**: State-of-the-art positional encoding
- **Squared ReLU Activation**: Improved training dynamics
- **RMS Normalization**: Efficient layer normalization
- **Standard PyTorch modules**: Built with `nn.Linear`, `F.scaled_dot_product_attention`

**Architecture**:
```
Input Tokens → Token Embedding → Transformer Blocks (Causal) → LM Head → Output Logits
                                  ↓
                           Rotary Position Embeddings
```

**Use Cases**:
- Standard language model training
- Autoregressive text generation
- Next-token prediction tasks

**Example Usage**:
```python
from models.gpt_causal import GPTCausal

model = GPTCausal(
    vocab_size=50304,
    n_embd=768,
    n_layer=12,
    n_head=12,
    dropout=0.1,
    max_seq_len=2048
)

# Training
logits, loss = model(tokens, targets=targets)

# Generation
generated = model.generate(prompt_tokens, max_new_tokens=100, temperature=0.8)
```

### 2. `flow_matching_transformer.py` - Bidirectional Flow Matching Transformer

**Purpose**: Flow matching training with time-conditioned bidirectional attention.

**Key Features**:
- **Bidirectional Attention**: Full attention across all tokens (no causal masking)
- **Time Conditioning**: Timestep embeddings concatenated at sequence start
- **Sinusoidal Time Embeddings**: Learned from flow matching timesteps
- **Squared ReLU Activation**: Improved training dynamics
- **RMS Normalization**: Efficient layer normalization
- **Standard PyTorch modules**: Built with `nn.Linear`, `F.scaled_dot_product_attention`

**Architecture**:
```
Timestep t → Time Embedding ─┐
                             ├→ Concatenate → Transformer Blocks (Bidirectional) → Remove Time Token → LM Head → Output Logits
Input Tokens → Token Embedding ─┘
```

**Use Cases**:
- Flow matching / continuous normalizing flows training
- Diffusion-style discrete token modeling
- Non-autoregressive generation

**Example Usage**:
```python
from models.flow_matching_transformer import FlowMatchingTransformer

model = FlowMatchingTransformer(
    vocab_size=50304,
    n_embd=768,
    n_layer=12,
    n_head=12,
    dropout=0.1,
    max_seq_len=2048,
    time_embed_dim=256
)

# Training (flow matching)
import torch
t = torch.rand(batch_size)  # Timesteps in [0, 1]
xt = ...  # Noised tokens (batch, seq_len, vocab_size)
targets = ...  # Target distributions

logits, loss = model(t, xt, targets=targets)

# Sampling
logits = model.sample(t, xt, temperature=1.0)
```

## Key Differences

| Feature | GPT Causal | Flow Matching Transformer |
|---------|-----------|---------------------------|
| **Attention Type** | Causal (autoregressive) | Bidirectional (full) |
| **Position Encoding** | Rotary Embeddings (RoPE) | None (uses time conditioning) |
| **Time Conditioning** | No | Yes (timestep embedding) |
| **Input Format** | Token indices or one-hot | Soft distributions (one-hot/soft labels) |
| **Training Objective** | Next-token prediction | Flow matching |
| **Generation** | Autoregressive | Iterative refinement |

## Model Configurations

Both models support flexible configurations through constructor arguments:

### Common Parameters
- `vocab_size`: Vocabulary size (required)
- `n_embd`: Embedding dimension (default: 768)
- `n_layer`: Number of transformer layers (default: 12)
- `n_head`: Number of attention heads (default: 12)
- `dropout`: Dropout probability (default: 0.0)
- `max_seq_len`: Maximum sequence length (default: 2048)
- `loss_func`: Custom loss function (default: `F.cross_entropy`)

### Flow Matching Specific
- `time_embed_dim`: Dimension for time frequency embeddings (default: 256)

## Integration with Config Files

To use these models in your training config, specify the model class:

```yaml
model:
  type: "FlowMatchingTransformer"  # or "GPTCausal"
  vocab_size: 50304
  n_embd: 768
  n_layer: 12
  n_head: 12
  dropout: 0.1
  max_seq_len: 512
```

## References

- **Rotary Position Embeddings**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- **Flow Matching**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- **Squared ReLU**: [Primer: Searching for Efficient Transformers](https://arxiv.org/abs/2109.08668)
