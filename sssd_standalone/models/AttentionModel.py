import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention module"""

    def __init__(self, d_model, num_heads=8, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (seq_len, batch_size, d_model)
        """
        seq_len, batch_size, d_model = x.shape

        # Linear projections: (seq_len, batch, d_model) -> (seq_len, batch, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        # (seq_len, batch, d_model) -> (seq_len, batch, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(seq_len, batch_size, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        K = K.view(seq_len, batch_size, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        V = V.view(seq_len, batch_size, self.num_heads, self.d_k).permute(1, 2, 0, 3)

        # Scaled dot-product attention
        # Q @ K^T: (batch, num_heads, seq_len, d_k) @ (batch, num_heads, d_k, seq_len) -> (batch, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, d_k) -> (batch, num_heads, seq_len, d_k)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back
        # (batch, num_heads, seq_len, d_k) -> (seq_len, batch, num_heads, d_k) -> (seq_len, batch, d_model)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(seq_len, batch_size, d_model)

        # Final linear projection
        output = self.W_o(attn_output)

        return output


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""

    def __init__(self, d_model, d_ff=None, dropout=0.0):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Output tensor of shape (seq_len, batch_size, d_model)
        """
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class AttentionLayer(nn.Module):
    """
    Full Attention Layer that can be used as a drop-in replacement for S4Layer.
    Maintains the same interface as S4Layer for compatibility with SSSD_ECG.
    """

    def __init__(self, features, lmax=None, num_heads=8, d_ff=None, dropout=0.0, layer_norm=True):
        """
        Args:
            features: Number of input/output features (d_model)
            lmax: Maximum sequence length (not used for attention, kept for compatibility)
            num_heads: Number of attention heads
            d_ff: Dimension of feedforward network (default: 4 * features)
            dropout: Dropout rate
            layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.features = features
        self.num_heads = num_heads

        # Multi-head self-attention
        self.attention = MultiHeadAttention(features, num_heads=num_heads, dropout=dropout)

        # Position-wise feedforward
        self.feedforward = PositionwiseFeedForward(features, d_ff=d_ff, dropout=dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(features) if layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(features) if layer_norm else nn.Identity()

        # Dropout
        self.dropout1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (seq_len, batch_size, features)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (seq_len, batch_size, features)
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(x, mask=mask)
        attn_output = self.dropout1(attn_output)
        x = x + attn_output  # Residual connection
        x = self.norm1(x)

        # Feedforward with residual connection and layer norm
        ff_output = self.feedforward(x)
        ff_output = self.dropout2(ff_output)
        x = x + ff_output  # Residual connection
        x = self.norm2(x)

        return x


class BidirectionalAttentionLayer(nn.Module):
    """
    Bidirectional Attention Layer that processes sequences in both directions.
    Similar to S4Layer with bidirectional=True.
    """

    def __init__(self, features, lmax=None, num_heads=8, d_ff=None, dropout=0.0, layer_norm=True):
        """
        Args:
            features: Number of input/output features (d_model)
            lmax: Maximum sequence length (not used for attention, kept for compatibility)
            num_heads: Number of attention heads
            d_ff: Dimension of feedforward network (default: 4 * features)
            dropout: Dropout rate
            layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.features = features

        # Forward and backward attention layers
        self.forward_layer = AttentionLayer(
            features, lmax=lmax, num_heads=num_heads,
            d_ff=d_ff, dropout=dropout, layer_norm=False
        )
        self.backward_layer = AttentionLayer(
            features, lmax=lmax, num_heads=num_heads,
            d_ff=d_ff, dropout=dropout, layer_norm=False
        )

        # Combine forward and backward outputs
        self.combine = nn.Linear(features * 2, features)

        # Layer normalization
        self.norm = nn.LayerNorm(features) if layer_norm else nn.Identity()

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (seq_len, batch_size, features)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (seq_len, batch_size, features)
        """
        # Forward direction
        forward_output = self.forward_layer(x, mask=mask)

        # Backward direction (reverse the sequence)
        x_reversed = torch.flip(x, dims=[0])
        backward_output = self.backward_layer(x_reversed, mask=mask)
        backward_output = torch.flip(backward_output, dims=[0])

        # Combine forward and backward outputs
        combined = torch.cat([forward_output, backward_output], dim=-1)
        output = self.combine(combined)
        output = self.norm(output)

        return output


def get_attention_layer(features, lmax=None, num_heads=8, d_ff=None, dropout=0.0,
                        bidirectional=True, layer_norm=True):
    """
    Factory function to create an attention layer.

    Args:
        features: Number of input/output features
        lmax: Maximum sequence length (kept for compatibility with S4Layer)
        num_heads: Number of attention heads
        d_ff: Dimension of feedforward network
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional attention
        layer_norm: Whether to use layer normalization

    Returns:
        AttentionLayer or BidirectionalAttentionLayer
    """
    if bidirectional:
        return BidirectionalAttentionLayer(
            features, lmax=lmax, num_heads=num_heads,
            d_ff=d_ff, dropout=dropout, layer_norm=layer_norm
        )
    else:
        return AttentionLayer(
            features, lmax=lmax, num_heads=num_heads,
            d_ff=d_ff, dropout=dropout, layer_norm=layer_norm
        )
