# Captioning model variant that also returns cross-attention weights.
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
from torch.nn import (
    Module,
    MultiheadAttention,
    Linear,
    Dropout,
    LayerNorm,
)
from typing import Optional, Any, Union, Callable


# Custom decoder layer exposes encoder-decoder attention for visualization.
class TransformerDecoderLayer(Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs,
        )
        self.multihead_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs,
        )
        self.linear1 = Linear(
            d_model,
            dim_feedforward,
            **factory_kwargs,
        )
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(
            dim_feedforward,
            d_model,
            **factory_kwargs,
        )

        self.norm_first = norm_first
        self.norm1 = LayerNorm(
            d_model,
            eps=layer_norm_eps,
            **factory_kwargs,
        )
        self.norm2 = LayerNorm(
            d_model,
            eps=layer_norm_eps,
            **factory_kwargs,
        )
        self.norm3 = LayerNorm(
            d_model,
            eps=layer_norm_eps,
            **factory_kwargs,
        )
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                tgt_mask,
                tgt_key_padding_mask,
            )
            temp = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
            )
            x = temp + self._ff_block(self.norm3(temp))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x,
                    tgt_mask,
                    tgt_key_padding_mask,
                )
            )
            temp, weights = self._mha_block(
                x,
                memory,
                memory_mask,
                memory_key_padding_mask,
            )
            x = self.norm2(x + temp)
            x = self.norm3(x + self._ff_block(x))

        return x, weights

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x, weights = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        return self.dropout2(x), weights

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(
            self.dropout(
                self.activation(self.linear1(x))
            )
        )
        return self.dropout3(x)


# Stack custom layers while retaining the final layer's attention map.
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
            output, temp = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output, temp


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


# Fixed sinusoidal positions for caption token embeddings.
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


# ViT encoder and custom decoder used by the attention visualization.
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
        # Project ViT patch features into decoder memory.
        memory = self.backbone(samples)
        memory = self.linear_proj(memory)
        memory = memory.permute(1, 0, 2)
        N, LEN = target.shape
        # Apply causal masking so decoding cannot inspect future tokens.
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
