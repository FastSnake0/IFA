import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.utils import load_image
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from PIL import Image
from io import BytesIO

from typing import Optional

MODEL_CACHE = {}
IP_ADAPTER_PATH = "h94/IP-Adapter"


def patch_kv(pipe, embeds: torch.Tensor, index: int = 0):
    if embeds.dim() == 2:
        embeds = embeds.unsqueeze(0)  # [1, seq_len, dim]

    def attention_kv_hook(module, args, kwargs):
        _, query, key, value, *_ = args
        return query, embeds, embeds

    count = 0
    for name, module in pipe.unet.named_modules():
        if "attn2" in name and hasattr(module, "_attention"):
            if count == index:
                module._attention = attention_kv_hook.__get__(module)
                break
            count += 1


def get_pipeline(model_name: str):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    if "xl" in model_name.lower():
        pipe = StableDiffusionXLPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

    pipe.enable_xformers_memory_efficient_attention()
    MODEL_CACHE[model_name] = pipe
    return pipe


def apply_ip_adapter(pipe, image: Image.Image, scale: float):
    pipe.load_ip_adapter(IP_ADAPTER_PATH, subfolder="models", weight_name="ip-adapter_sd15.bin")
    image = image.convert("RGB").resize((512, 512))
    pipe.set_ip_adapter_scale(scale)
    return pipe, image


def generate_image(
    model: str,
    pos_prompt: str,
    ng_prompt: Optional[str] = None,
    ip: Optional[dict] = None,
    inject: Optional[dict] = None
) -> BytesIO:
    pipe = get_pipeline(model)

    if ip is not None:
        image = load_image(ip["image"])
        scale = float(ip.get("scale", 1.0))
        pipe, image = apply_ip_adapter(pipe, image, scale)
        image_output = pipe(pos_prompt, negative_prompt=ng_prompt, ip_adapter_image=image).images[0]

    elif inject is not None:
        prompt = inject["prompt"]
        scale = float(inject.get("scale", 1.0))
        index = int(inject.get("index", 0))

        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(pipe.device)
        embeds = text_encoder(input_ids)[0] * scale  # [1, seq, dim]

        patch_kv(pipe, embeds, index)
        image_output = pipe(pos_prompt, negative_prompt=ng_prompt).images[0]

    else:
        image_output = pipe(pos_prompt, negative_prompt=ng_prompt).images[0]

    buf = BytesIO()
    image_output.save(buf, format="PNG")
    buf.seek(0)
    return buf