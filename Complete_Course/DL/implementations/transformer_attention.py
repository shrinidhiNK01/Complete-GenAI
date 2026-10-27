"""
Self-Attention, Multi-Head Attention, and Transformer Block in PyTorch.
This script implements the core mathematical components of the Transformer architecture.

Run: python Complete_Course/DL/implementations/transformer_attention.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        # q, k, v shapes: (batch_size, num_heads, seq_len, d_k)
        d_k = q.size(-1)
        
        # 1. Compute attention scores: Q * K^T
        # We transpose the last two dimensions of K to do matrix multiplication: (seq_len, d_k) x (d_k, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 2. Apply optional attention mask (e.g. causal mask for text generation)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 3. Apply Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # 4. Multiply attention weights by Value vectors: weights * V
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Projection layers for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Output projection layer
        self.out_linear = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Project inputs to Q, K, V and split into multiple heads
        # Shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k) -> transpose to (batch_size, num_heads, seq_len, d_k)
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply scaled dot-product attention
        x, attn_weights = self.attention(q, k, v, mask=mask)
        
        # 3. Concatenate heads back together
        # Transpose: (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k)
        # Reshape (concat): -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. Final output projection
        return self.out_linear(x), attn_weights


class TransformerBlock(nn.Module):
    """
    A single standard Transformer Block containing Multi-Head Attention,
    Feed-Forward Network, Residual connections, and Layer Normalization.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Position-wise Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN architecture (standard in modern models like GPT)
        # 1. Attention + Residual connection
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x, mask=mask)
        x = x + self.dropout(attn_out)
        
        # 2. Feed-Forward Network + Residual connection
        norm_x2 = self.norm2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + self.dropout(ffn_out)
        
        return x


def main():
    # Define sequence dimensions
    batch_size = 2
    seq_len = 5
    d_model = 64  # Hidden dimension size
    num_heads = 4
    d_ff = 256    # Dimension of Feed-Forward Hidden Layer
    
    # 1. Create dummy input sequence (e.g. word embeddings)
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input Sequence Shape: {x.shape} (Batch Size={batch_size}, Seq Len={seq_len}, Embed Dim={d_model})")
    
    # 2. Create standard multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)
    attn_out, weights = mha(x, x, x)
    print(f"Multi-Head Attention Output Shape: {attn_out.shape}")
    print(f"Attention Weights Shape: {weights.shape} (Batch, Heads, Query-words, Key-words)\n")

    # 3. Create full Transformer Block
    block = TransformerBlock(d_model, num_heads, d_ff)
    block_out = block(x)
    print(f"Transformer Block Output Shape: {block_out.shape}")
    print("Transformer Block executed successfully!")

if __name__ == '__main__':
    main()
