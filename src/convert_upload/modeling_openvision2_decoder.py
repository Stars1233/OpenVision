"""PyTorch reference implementation of the OpenVision2 caption decoder.

The OpenVision2 generative model is: a ViT vision encoder (released separately as
the `*-vision-only` open_clip checkpoint) whose patch tokens condition a small
autoregressive text decoder trained with a captioning loss.

This decoder is a **prefix-LM / concat-fusion** transformer (NOT a CoCa-style
cross-attention decoder):

    text_embeds  = Embed(text_tokens)                      # no positional embedding
    image_embeds = image_projection(vit_patch_tokens)      # Linear, no bias
    x            = concat([image_embeds, text_embeds], 1)  # [B, N_img + L, width]
    x            = prefix_lm_transformer(x)                 # image=bidirectional prefix,
                                                            # text=causal, text->image=full,
                                                            # image-/->text
    logits       = lm_head( ln_final( x[:, N_img:] ) )     # over text positions

Faithful details that matter for numerical parity with the JAX model:
  * LayerNorm eps = 1e-6 (flax default), pre-LN blocks.
  * MLP activation = gelu(tanh approximation).
  * attention scale = head_dim ** -0.5.
  * NO positional embedding on the text stream.
  * vision patch tokens are the pre-final-norm tokens, cls excluded
    (open_clip vision model with output_tokens=True returns exactly these).
"""
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OpenVision2TextDecoderConfig:
    width: int = 1024          # decoder hidden width (H decoder -> 1024)
    depth: int = 24            # number of transformer blocks
    num_heads: int = 16
    mlp_dim: int = 4096
    vocab_size: int = 32000
    vision_width: int = 1280   # ViT token dim feeding image_projection (H ViT -> 1280)
    layer_norm_eps: float = 1e-6
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2


class _Mlp(nn.Module):
    def __init__(self, width: int, mlp_dim: int):
        super().__init__()
        self.c_fc = nn.Linear(width, mlp_dim)
        self.c_proj = nn.Linear(mlp_dim, width)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class _Attention(nn.Module):
    """MHA with a fused in_proj (q,k,v) and an explicit boolean attend-mask."""

    def __init__(self, width: int, num_heads: int):
        super().__init__()
        assert width % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.scale = self.head_dim ** -0.5
        self.in_proj_weight = nn.Parameter(torch.empty(3 * width, width))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * width))
        self.out_proj = nn.Linear(width, width)

    def forward(self, x, attend_mask):
        # attend_mask: [L, L] bool, True = keep, False = -inf
        B, L, D = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        H, hd = self.num_heads, self.head_dim
        q = q.view(B, L, H, hd).transpose(1, 2)   # [B,H,L,hd]
        k = k.view(B, L, H, hd).transpose(1, 2)
        v = v.view(B, L, H, hd).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,L,L]
        attn = attn.masked_fill(~attend_mask[None, None], float("-inf"))
        attn = attn.softmax(dim=-1)
        out = attn @ v                                  # [B,H,L,hd]
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class _Block(nn.Module):
    def __init__(self, cfg: OpenVision2TextDecoderConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.width, eps=cfg.layer_norm_eps)
        self.attn = _Attention(cfg.width, cfg.num_heads)
        self.ln_2 = nn.LayerNorm(cfg.width, eps=cfg.layer_norm_eps)
        self.mlp = _Mlp(cfg.width, cfg.mlp_dim)

    def forward(self, x, attend_mask):
        x = x + self.attn(self.ln_1(x), attend_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class OpenVision2TextDecoder(nn.Module):
    def __init__(self, cfg: OpenVision2TextDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.image_projection = nn.Linear(cfg.vision_width, cfg.width, bias=False)
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(cfg.depth)])
        self.ln_final = nn.LayerNorm(cfg.width, eps=cfg.layer_norm_eps)
        self.lm_head = nn.Linear(cfg.width, cfg.vocab_size, bias=False)

    @staticmethod
    def _prefix_lm_mask(li: int, lt: int, device) -> torch.Tensor:
        """[l, l] bool attend-mask. image(:li) bidirectional prefix; text(li:)
        causal + attends all image; image does NOT attend text."""
        l = li + lt
        mask = torch.zeros(l, l, dtype=torch.bool, device=device)
        mask[:li, :li] = True                              # image <-> image
        mask[li:, :li] = True                              # text  -> image
        text_causal = torch.tril(torch.ones(lt, lt, dtype=torch.bool, device=device))
        mask[li:, li:] = text_causal                       # text causal self
        return mask

    def forward(self, image_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """image_tokens: [B, N_img, vision_width]; text_tokens: [B, L] (input side).
        Returns logits [B, L, vocab] (next-token logits at each text position)."""
        text_embeds = self.token_embedding(text_tokens)          # [B, L, width]
        image_embeds = self.image_projection(image_tokens)       # [B, N_img, width]
        li, lt = image_embeds.shape[1], text_embeds.shape[1]
        x = torch.cat([image_embeds, text_embeds], dim=1)        # [B, li+lt, width]
        mask = self._prefix_lm_mask(li, lt, x.device)
        for blk in self.blocks:
            x = blk(x, mask)
        x = x[:, li:]                                            # text positions
        x = self.ln_final(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, image_tokens: torch.Tensor, max_len: int = 64,
                 bos_id: Optional[int] = None, eos_id: Optional[int] = None) -> torch.Tensor:
        """Greedy autoregressive caption generation. image_tokens: [B, N, vision_width]."""
        bos_id = self.cfg.bos_id if bos_id is None else bos_id
        eos_id = self.cfg.eos_id if eos_id is None else eos_id
        B = image_tokens.shape[0]
        device = image_tokens.device
        seq = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len):
            logits = self.forward(image_tokens, seq)             # [B, cur_len, vocab]
            nxt = logits[:, -1].argmax(dim=-1)                   # [B]
            nxt = torch.where(done, torch.full_like(nxt, self.cfg.pad_id), nxt)
            seq = torch.cat([seq, nxt[:, None]], dim=1)
            done = done | (nxt == eos_id)
            if bool(done.all()):
                break
        return seq
