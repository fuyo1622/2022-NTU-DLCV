# Transformer captioning model used for HW3 training and prediction.
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
from typing import Optional, List
from torch import Tensor
import numpy as np
import copy
import math


# Stack independent copies of PyTorch's standard decoder layer.
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        ff_dim,
        nb_heads,
        num_layers,
        drop_val=0.1,
        norm=None,
    ):
        super(TransformerDecoder, self).__init__()
        self.decoder_layer = TransformerDecoderLayer(
            d_model=in_dim,
            nhead=nb_heads,
            dim_feedforward=ff_dim,
        )
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(self.decoder_layer)
                for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


# Generic feed-forward projection retained by the captioning architecture.
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip(
                [input_dim] + h,
                h + [output_dim],
            )
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = (
                F.relu(layer(x))
                if i < self.num_layers - 1
                else layer(x)
            )
        return x


# Fixed sinusoidal positions for the autoregressive caption tokens.
class pPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(pPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


# Combine a ViT image encoder with a from-scratch text decoder.
class Caption(nn.Module):
    def __init__(
        self,
        in_dim,
        hd_dim,
        ff_dim,
        nb_heads,
        num_layers,
        seq_length,
        nb_tokens,
        padding_idx,
        backbone,
        device,
    ):
        super().__init__()
        self.device = device
        self.backbone = backbone
        self.position_encoder = pPositionalEncoding(
            hd_dim,
            dropout=0,
            max_len=seq_length,
        )
        self.decoder = TransformerDecoder(
            hd_dim,
            ff_dim,
            nb_heads,
            num_layers,
        )
        self.token_embedder = nn.Embedding(
            nb_tokens,
            hd_dim,
            padding_idx,
        )
        self.generator = nn.Linear(hd_dim, nb_tokens)
        self.mlp = MLP(hd_dim, 512, nb_tokens, 3)
        self.position_embedding = nn.Embedding(seq_length, hd_dim)
        self.linear_proj = nn.Linear(768, hd_dim, bias=False)
        self.fc = nn.Linear(hd_dim, nb_tokens)

    def forward(self, samples, target, target_mask):
        # Convert image patches into the decoder's cross-attention memory.
        memory = self.backbone(samples)
        memory = self.linear_proj(memory)
        memory = memory.permute(1, 0, 2)
        N, LEN = target.shape
        # Prevent each token from attending to future caption positions.
        tgt_mask = (
            nn.Transformer.generate_square_subsequent_mask(
                target.size(-1)
            ).to(self.device)
        )
        target = self.token_embedder(target)
        target = self.position_encoder(target)
        target = target.permute(1, 0, 2)

        out = self.decoder(
            target,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=target_mask,
        ).permute(1, 0, 2)
        out = self.fc(out)

        return out
