import torch
from torch import nn
from models.transformers.attn import MultiHeadSelfAttention

# encoder
class TransformerEncoderLayer(nn.Module):
  """ Transformer encoder layer """
  def __init__(self, d_model=256, num_heads=8, ffn_size=1024):
     super().__init__()

     self.attn = MultiHeadSelfAttention(
          d_model=d_model,
          num_heads=num_heads
     )

     self.ffn = nn.Sequential(
          nn.Linear(d_model, ffn_size),
          nn.ReLU(),
          nn.Linear(ffn_size, d_model)
     )

     self.norm1 = nn.LayerNorm(d_model)
     self.norm2 = nn.LayerNorm(d_model)

  def forward(self, x, pos=None):
     x = x + self.attn(x, x, x, pos, pos)
     x = self.norm1(x)

     # Feed-forward network
     x = x + self.ffn(x)
     x = self.norm2(x)
     return x
     
class TransformerEncoder(nn.Module):
   """ Stack all encoder layers to build an encoder """
   def __init__(self, num_layers=6, d_model=256, num_heads=8, ffn_size=1024):
       super().__init__()

       self.layers = nn.ModuleList([
          TransformerEncoderLayer(
             d_model=d_model,
             num_heads=num_heads,
             ffn_size=ffn_size
          ) for _ in range(num_layers)
        ]
      )

   def forward(self, x, pos=None):
      for layer in self.layers:
          x = layer(x, pos)
      return x
      
# decoder
class TransformerDecoderLayer(nn.Module):
   """ Decoder layer of DETR for Cross-Attention """
   def __init__(self, d_model=256, num_heads=8, ffn_size=1024):
     super().__init__()

     self.attn = MultiHeadSelfAttention(
          d_model=d_model,
          num_heads=num_heads
     )

     self.ffn = nn.Sequential(
          nn.Linear(d_model, ffn_size),
          nn.ReLU(),
          nn.Linear(ffn_size, d_model)
     )

     self.norm1 = nn.LayerNorm(d_model)
     self.norm2 = nn.LayerNorm(d_model)

   def forward(self, query, key, value, query_pos=None, key_pos=None):
       x = query + self.attn(query, key, value, query_pos, key_pos)
       x = self.norm1(x)

       # Feed-forward network
       x = x + self.ffn(x)
       x = self.norm2(x)
       return x
       
class TransformerDecoder(nn.Module):
   """ Stack all encoder layers to build an encoder """
   def __init__(self, num_layers=6, d_model=256, num_heads=8, ffn_size=1024):
       super().__init__()

       self.layers = nn.ModuleList([
          TransformerDecoderLayer(
             d_model=d_model,
             num_heads=num_heads,
             ffn_size=ffn_size
          ) for _ in range(num_layers)
        ]
      )

   def forward(self, query, key, value, query_pos=None, key_pos=None):
      for layer in self.layers:
          x = layer(query, key, value, query_pos, key_pos)
      return x