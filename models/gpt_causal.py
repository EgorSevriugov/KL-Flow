"""
GPT-style Causal Transformer with Rotary Positional Embeddings
Uses standard PyTorch modules for autoregressive language modeling
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for transformer models.
    Based on: https://arxiv.org/abs/2104.09864
    """
    def __init__(self, dim, max_seq_len=8192, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for cos/sin values
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
    
    def _compute_cos_sin(self, seq_len, device, dtype):
        """Compute and cache cos/sin values for given sequence length"""
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.cos_cached = emb.cos().to(dtype)
            self.sin_cached = emb.sin().to(dtype)
        return self.cos_cached, self.sin_cached
    
    def forward(self, x):
        """
        Apply rotary embeddings to input tensor
        Args:
            x: Input tensor of shape (batch, seq_len, n_head, head_dim)
        Returns:
            Rotated tensor of same shape
        """
        seq_len = x.shape[1]
        cos, sin = self._compute_cos_sin(seq_len, x.device, x.dtype)
        
        # Reshape for broadcasting
        cos = cos[None, :, None, :]  # (1, seq_len, 1, dim)
        sin = sin[None, :, None, :]
        
        # Split into two halves and apply rotation
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated


class MultiHeadAttentionWithRoPE(nn.Module):
    """
    Multi-head attention with Rotary Position Embeddings and causal masking
    """
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # QKV projections
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        # Rotary embeddings
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        
        self.dropout = dropout
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Input tensor (batch, seq_len, n_embd)
            attention_mask: Optional attention mask
        Returns:
            Output tensor (batch, seq_len, n_embd)
        """
        B, T, C = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim)
        
        # Apply RMS normalization to Q and K
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        
        # Apply rotary embeddings
        q = self.rotary(q)
        k = self.rotary(k)
        
        # Transpose for attention computation (batch, n_head, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(attention_mask is None)  # Use causal masking if no mask provided
        )
        
        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        
        return y


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with squared ReLU activation
    """
    def __init__(self, n_embd, dropout=0.0, expansion_factor=4):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, expansion_factor * n_embd, bias=False)
        self.fc2 = nn.Linear(expansion_factor * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Zero init output projection
        self.fc2.weight.data.zero_()
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x).square()  # Squared ReLU activation
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-normalization and residual connections
    """
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttentionWithRoPE(n_embd, n_head, dropout)
        self.ff = FeedForward(n_embd, dropout)
    
    def forward(self, x, attention_mask=None):
        # Pre-norm with residual connection
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)), attention_mask)
        x = x + self.ff(F.rms_norm(x, (x.size(-1),)))
        return x


class GPTCausal(nn.Module):
    """
    GPT-style Causal Transformer with Rotary Position Embeddings
    
    Args:
        vocab_size: Size of vocabulary
        n_embd: Embedding dimension
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        dropout: Dropout probability
        max_seq_len: Maximum sequence length
    """
    def __init__(
        self,
        vocab_size,
        n_embd=768,
        n_layer=12,
        n_head=12,
        dropout=0.0,
        max_seq_len=2048,
        loss_func=None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.loss_func = loss_func
        
        # Token embeddings (using Linear instead of Embedding for consistency)
        self.token_emb = nn.Linear(vocab_size, n_embd, bias=False)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout)
            for _ in range(n_layer)
        ])
        
        # Output head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, targets=None, attention_mask=None, return_logits=True):
        """
        Forward pass
        
        Args:
            x: Input token indices (batch, seq_len) or one-hot (batch, seq_len, vocab_size)
            targets: Optional target tokens for loss computation
            attention_mask: Optional attention mask
            return_logits: Whether to return logits
            
        Returns:
            logits: Output logits (if return_logits=True)
            loss: Cross-entropy loss (if targets provided)
        """
        # Get token embeddings
        if x.dim() == 2:
            # Convert indices to one-hot
            x = F.one_hot(x, num_classes=self.vocab_size).float()
        
        x = self.token_emb(x)  # (batch, seq_len, n_embd)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer norm
        x = F.rms_norm(x, (x.size(-1),))
        
        # Compute logits
        logits = self.lm_head(x) if return_logits or targets is not None else None
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            if logits is None:
                logits = self.lm_head(x)
            
            logits = logits.float()  # Use fp32 for loss computation
            
            loss_func = self.loss_func if self.loss_func is not None else F.cross_entropy
            
            if len(targets.shape) == 2:
                # Standard cross-entropy with token indices
                loss = loss_func(logits.view(-1, self.vocab_size), targets.view(-1))
            else:
                # Targets are distributions (e.g., one-hot or soft labels)
                loss = loss_func(logits.view(-1, self.vocab_size), targets.view(-1, self.vocab_size))
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation
        
        Args:
            idx: Starting token indices (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Optional top-k filtering
            
        Returns:
            Generated token indices (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            # Forward pass
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
