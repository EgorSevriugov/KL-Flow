"""
Flow Matching Bidirectional Transformer with Time Conditioning
Uses standard PyTorch modules for flow matching training
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using sinusoidal embeddings.
    Based on Diffusion/Flow Matching literature.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: A 1-D or 2-D Tensor of indices (batch,) or (batch, seq_len)
            dim: The dimension of the output
            max_period: Controls the minimum frequency of the embeddings
            
        Returns:
            Positional embeddings of shape (batch, dim) or (batch, seq_len, dim)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        # Handle both (batch,) and (batch, seq_len) shapes
        if t.dim() == 1:
            args = t.float().unsqueeze(-1) * freqs
        else:
            args = torch.einsum("...i,i->...i", t.unsqueeze(-1).float(), freqs)
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        
        return embedding

    def forward(self, t):
        """
        Args:
            t: Timestep tensor (batch,) or (batch, seq_len)
        Returns:
            Time embeddings (batch, hidden_size) or (batch, seq_len, hidden_size)
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class MultiHeadAttentionBidirectional(nn.Module):
    """
    Multi-head bidirectional attention (no causal masking)
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
        
        # Zero init output projection
        self.out_proj.weight.data.zero_()
        
        self.dropout = dropout
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Input tensor (batch, seq_len, n_embd)
            attention_mask: Optional attention mask (batch, seq_len, seq_len)
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
        
        # Transpose for attention computation (batch, n_head, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention WITHOUT causal masking (bidirectional)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False  # Bidirectional attention
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
    Bidirectional transformer block with pre-normalization and residual connections
    """
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttentionBidirectional(n_embd, n_head, dropout)
        self.ff = FeedForward(n_embd, dropout)
    
    def forward(self, x, attention_mask=None):
        # Pre-norm with residual connection
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)), attention_mask)
        x = x + self.ff(F.rms_norm(x, (x.size(-1),)))
        return x


class FlowMatchingTransformer(nn.Module):
    """
    Bidirectional Transformer for Flow Matching with Time Conditioning
    
    The time embedding is concatenated at the beginning of the sequence,
    allowing the model to condition on the timestep throughout processing.
    
    Args:
        vocab_size: Size of vocabulary
        n_embd: Embedding dimension
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        dropout: Dropout probability
        max_seq_len: Maximum sequence length (not including time token)
        time_embed_dim: Dimension for frequency embeddings of time
    """
    def __init__(
        self,
        vocab_size,
        n_embd=768,
        n_layer=12,
        n_head=12,
        dropout=0.0,
        max_seq_len=2048,
        time_embed_dim=256,
        loss_func=None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.loss_func = loss_func
        
        # Time embedder
        self.time_embedder = TimestepEmbedder(n_embd, time_embed_dim)
        
        # Token embeddings (using Linear instead of Embedding for consistency with one-hot inputs)
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
    
    def forward(self, t, xt, targets=None, attention_mask=None, return_logits=True):
        """
        Forward pass with time conditioning
        
        Args:
            t: Timestep (batch,) - scalar timestep for each sample in batch
            xt: Input tokens (batch, seq_len, vocab_size) - one-hot or soft distributions
            targets: Optional target tokens for loss computation
            attention_mask: Optional attention mask
            return_logits: Whether to return logits
            
        Returns:
            logits: Output logits (if return_logits=True)
            loss: Loss value (if targets provided)
        """
        B, T, V = xt.shape
        
        # Get time embeddings (batch, n_embd)
        t_emb = self.time_embedder(t)
        
        # Get token embeddings (batch, seq_len, n_embd)
        x = self.token_emb(xt)
        
        # Concatenate time embedding at the beginning of sequence
        # (batch, 1, n_embd) + (batch, seq_len, n_embd) -> (batch, seq_len+1, n_embd)
        x = torch.cat([t_emb.unsqueeze(1), x], dim=1)
        
        # Extend attention mask if provided
        if attention_mask is not None:
            # Add column for time token
            time_mask = torch.ones(B, 1, attention_mask.size(-1) + 1, device=attention_mask.device)
            attention_mask = torch.cat([time_mask, attention_mask], dim=1)
        
        # Apply transformer blocks (bidirectional attention)
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer norm
        x = F.rms_norm(x, (x.size(-1),))
        
        # Remove time token from output
        x = x[:, 1:]  # (batch, seq_len, n_embd)
        
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
    
    @torch.no_grad()
    def sample(self, t, xt, temperature=1.0, top_k=None):
        """
        Sample from the model (for flow matching sampling/generation)
        
        Args:
            t: Timestep (batch,)
            xt: Current state (batch, seq_len, vocab_size)
            temperature: Sampling temperature
            top_k: Optional top-k filtering
            
        Returns:
            Sampled logits or probabilities
        """
        logits, _ = self.forward(t, xt, return_logits=True)
        logits = logits / temperature
        
        # Optional top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            logits[logits < v[..., [-1]]] = -float('Inf')
        
        return logits
