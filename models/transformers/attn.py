import math
import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, query_pos=None, key_pos=None):
        if query_pos is not None:
           query = query + query_pos

        if key_pos is not None:
           key = key + key_pos

        B, Nq, C = query.shape
        _, Nk, _ = key.shape

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # split into heads
        Q = Q.reshape(B, Nq, self.num_heads, self.head_dim)
        K = K.reshape(B, Nk, self.num_heads, self.head_dim)
        V = V.reshape(B, Nk, self.num_heads, self.head_dim)

        # reshape
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # perform attention
        scores = Q @ K.transpose(-2, -1) # QK^T
        scores = scores / math.sqrt(self.head_dim)
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        output = scores @ V
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(B, Nq, C)

        output = self.out_proj(output)
        return output
        
class PositionalEmbedding(nn.Module):
   """ Sine, Cosine embedding for positional encoding """
   def __init__(self, num_pos_feats=128, temperature=10000):
       super().__init__()

       self.num_pos_feat = num_pos_feats
       self.temp = temperature

   def forward(self, x):
       B, C, H, W = x.shape

       x_pos = torch.arange(W, device=x.device)
       y_pos = torch.arange(H, device=x.device)

       x_embed = x_pos[:, None].expand(H, W)
       y_embed = y_pos[:, None].expand(H, W)

       dim_t = torch.arange(self.num_pos_feat, device=x.device)

       dim_t = self.temp ** (
          2 * (dim_t // 2) / self.num_pos_feat
       )

       pos_x = x_embed[..., None] / dim_t
       pos_y = y_embed[..., None] / dim_t

       pos_x = torch.stack((
          pos_x[..., 0::2].sin(),
          pos_x[..., 1::2].cos()
          ),
          dim=-1
       ).flatten(-2)
       pos_y = torch.stack((
          pos_y[..., 0::2].sin(),
                pos_y[..., 1::2].cos()
          ),
          dim=-1
       ).flatten(-2)

       pos = torch.cat([pos_x, pos_y], dim=-1).permute(2, 0, 1)

       pos = pos.unsqueeze(0).flatten(2).transpose(1, 2)
       return pos

class PreTransform(nn.Module):
    """ Perform some transform before transformer block """
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
       x = self.proj(x)
       x = x.flatten(2)
       x = x.permute(0, 2, 1)
       return x