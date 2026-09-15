import timm
import torch
from torch import nn

from models.transformers.encoder_decoder import TransformerEncoder, TransformerDecoder
from models.transformers.attn import PreTransform, PositionalEmbedding
from models.head import DETRHead

def get_backbone(name: str = "resnet50", pretrained: bool = True):
    """ Strip the fc layer and return the backbone + its output channel count """
    model = timm.create_model(name, pretrained=pretrained)
    out_channels = model.num_features  # channel dim of final feature map, pre-pool/fc
    del model.fc, model.global_pool
    return model, out_channels

class DETR(nn.Module):
    """ End to end detr model putting all together """
    def __init__(
        self,
        num_classes,
        backbone_name="resnet50",
        pretrained=True,
        num_attn_heads=8,
        num_transformer_layers=6,
        d_model=256,
        ffn_size=1024,
        num_pos_features=128,
        scale_temperature=10000,
        num_query_size=100
    ):
        super().__init__()

        self.backbone, in_channels = get_backbone(backbone_name, pretrained) # backbone

        self.b4_transformer = PreTransform(in_channels, d_model) # 1x1 conv and flatten, reshape

        self.pos_embed = PositionalEmbedding(num_pos_features, scale_temperature) # positional embedding

        self.encoder = TransformerEncoder(num_transformer_layers, d_model, num_attn_heads, ffn_size) # transformer encoder

        self.query_embed = nn.Embedding(num_query_size, d_model) # query embed

        self.decoder = TransformerDecoder(num_transformer_layers, d_model, num_attn_heads, ffn_size) # transformer decoder

        self.head = DETRHead(num_classes, d_model, d_model, 3) # prediction head

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone.forward_features(x) # [B, 2048, 20, 20]

        feat = self.b4_transformer(x) # [B, 400, 256]
        pos = self.pos_embed(x) # [B, 400, 256]

        x = self.encoder(feat, pos) # [B, 400, 256]

        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        x = self.decoder(query, x, x, query, pos) # [B, 100, 256]

        # classification logits, bbox [cx, cy, w, h]
        return self.head(x)