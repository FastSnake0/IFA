import torch
import torch.nn as nn
from PIL import Image
from typing import List, Optional, Tuple, Union

from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import torch.nn.functional as F


from einops import rearrange
from einops.layers.torch import Rearrange
import math

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)

# -------------------------------------------------------------------
# 1. Класс адаптера: загрузка проекции, CLIP и весов IP-Adapter(Plus)
# -------------------------------------------------------------------
from safetensors.torch import safe_open
class IPAdapterModule:
    def __init__(
        self,
        adapter_ckpt: str,
        image_encoder_repo: str,
        device: Union[str, torch.device],
        num_tokens: int = 4,
        is_plus: bool = False,
        cross_attention_dim: int = 768,
        layers_to_patch: Optional[List[Union[int, str]]] = None,
    ):
        """
        adapter_ckpt: путь к файлу .safetensors
        image_encoder_repo: hf repo_id для CLIP vision
        device: 'cuda' или 'cpu'
        num_tokens: количество image-токенов
        is_plus: использовать IP-Adapter-Plus (Resampler) или простой ImageProjModel
        cross_attention_dim: размер эмбеддинга UNet
        layers_to_patch: список индексов или имён слоёв для патчинга; None = все
        """
        self.device = torch.device(device)
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.layers_to_patch = layers_to_patch

        # CLIP vision encoder
        self.clip = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_repo, projection_dim=cross_attention_dim
        ).to(self.device)
        self.processor = CLIPImageProcessor()

        # Image projection model
        if is_plus:
           
            self.proj = Resampler(
                dim=cross_attention_dim,
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=num_tokens,
                embedding_dim=self.clip.config.projection_dim,
                output_dim=cross_attention_dim,
                ff_mult=4,
            ).to(self.device)
        else:
            self.proj = ImageProjModel(
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=self.clip.config.projection_dim,
                clip_extra_context_tokens=num_tokens,
            ).to(self.device)

        # Load weights
        state_dict = {"image_proj": {}, "ip_adapter": {}}
        with safe_open(adapter_ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("image_proj."):
                    state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                elif key.startswith("ip_adapter."):
                    state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)

        self.proj.load_state_dict(state_dict["image_proj"])
        self.ip_state = state_dict["ip_adapter"]

    @torch.inference_mode()
    def get_image_tokens(
        self,
        pil_images: List[Image.Image],
    ) -> torch.FloatTensor:
        pix = self.processor(images=pil_images, return_tensors="pt").pixel_values.to(self.device)
        clip_emb = self.clip(pix).image_embeds
        tokens = self.proj(clip_emb)
        return tokens

    def patch_unet(
        self,
        unet: UNet2DConditionModel,
    ):
        

        # Collect cross-attn layers
        names = list(unet.attn_processors.keys())
        # Filter by layers_to_patch
        if self.layers_to_patch is not None:
            filtered = []
            for idx, name in enumerate(names):
                if isinstance(self.layers_to_patch[0], int):
                    if idx in self.layers_to_patch:
                        filtered.append(name)
                else:
                    if name in self.layers_to_patch:
                        filtered.append(name)
            target_names = set(filtered)
        else:
            target_names = set(names)

        procs = {}
        for idx, name in enumerate(names):
            if name not in target_names:
                # Default processor (no image conditioning)
                procs[name] = AttnProcessor()
                continue
            # Patch this layer
            if name.endswith("attn1.processor"):
                procs[name] = AttnProcessor()
            else:
                # Determine hidden_size
                if name.startswith("mid_block"):
                    hs = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    b = int(name.split(".")[1]); hs = list(reversed(unet.config.block_out_channels))[b]
                else:
                    b = int(name.split(".")[1]); hs = unet.config.block_out_channels[b]
                proc = IPAttnProcessor(
                    hidden_size=hs,
                    cross_attention_dim=unet.config.cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device)
                procs[name] = proc
        unet.set_attn_processor(procs)
        layers = nn.ModuleList(unet.attn_processors.values())
        layers.load_state_dict(self.ip_state)

    def set_scale(
        self,
        unet: UNet2DConditionModel,
        scale: float,
    ):
        for proc in unet.attn_processors.values():
            if hasattr(proc, 'scale'):
                proc.scale = scale

    def generate_batch(
        self,
        pipe: StableDiffusionPipeline,
        prompts: List[str],
        image_prompts: List[Optional[Image.Image]],
        masks: List[Optional[torch.Tensor]],
        sigma_range: Tuple[float, float] = (1.0, 0.0),
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        pipe = pipe.to(self.device)
        # Prepare image tokens
        tokens = []
        for img in image_prompts:
            if img is None:
                tokens.append(torch.zeros(1, self.num_tokens, pipe.unet.config.cross_attention_dim, device=self.device))
            else:
                tokens.append(self.get_image_tokens([img]))
        image_tokens = torch.cat(tokens, dim=0)
        # Generate
        generator = torch.manual_seed(seed) if seed is not None else None
        out = pipe(
            prompts,
            image_prompt_embeds=image_tokens,
            negative_prompt_embeds=torch.zeros_like(image_tokens),
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            cross_attention_kwargs={
                'mask': masks,
                'sigma_range': sigma_range,
            },
        )
        return out.images