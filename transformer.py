import torch
import torch.nn as nn
import os
from gpa import GroupedQueryAttention
from forward import FeedForward
from rmsnorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg, attn_type):
        super().__init__()
        self.attn_type = attn_type

        self.dtype = torch.bfloat16
        if cfg["torch_dtype"] != "bfloat16":
            print(f"unexpected dtype {self.dtype}")
            os._exit(-1);

        self.att = GroupedQueryAttention(
            d_in=cfg["hidden_size"],
            num_heads=cfg["num_attention_heads"],
            num_kv_groups=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            qk_norm=cfg["qk_norm"],
            query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
            dtype=self.dtype,
        )
        self.ff = FeedForward(cfg)
        self.input_layernorm = RMSNorm(cfg["hidden_size"], eps=1e-6)
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm(cfg["hidden_size"], eps=1e-6)
        self.post_feedforward_layernorm = RMSNorm(cfg["hidden_size"], eps=1e-6)

    def forward(
        self,
        x,
        mask_global,
        mask_local,
        cos_global,
        sin_global,
        cos_local,
        sin_local,
    ):
        # Shortcut connection for attention block
        shortcut = x
        x = self.input_layernorm(x)

        if self.attn_type == "sliding_attention":
            attn_mask = mask_local
            cos = cos_local
            sin = sin_local
        else:
            attn_mask = mask_global
            cos = cos_global
            sin = sin_global

        x_attn = self.att(x, attn_mask, cos, sin)
        x_attn = self.post_attention_layernorm(x_attn)
        x = shortcut + x_attn

        # Shortcut connection for feed forward block
        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn
        return x
