from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, Optional[float]], None]


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _load_pairs(dataset_path: str) -> List[Dict[str, str]]:
    root = Path(dataset_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"dataset_path not found: {root}")

    if root.is_file():
        suffix = root.suffix.lower()
        if suffix == ".jsonl":
            items: List[Dict[str, str]] = []
            base_dir = root.parent
            for line in root.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                image_path = str(obj.get("image_path") or obj.get("image") or "").strip()
                text = str(obj.get("text") or obj.get("caption") or "").strip()
                if not image_path:
                    continue
                img = Path(image_path)
                if not img.is_absolute():
                    img = (base_dir / img).resolve()
                items.append({"image_path": str(img), "text": text})
            if not items:
                raise ValueError(f"No samples found in jsonl: {root}")
            return items

        if suffix == ".json":
            payload = json.loads(root.read_text(encoding="utf-8"))
            base_dir = root.parent
            rows: Iterable[Dict[str, Any]]
            if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                rows = payload["data"]
            elif isinstance(payload, list):
                rows = payload
            else:
                raise ValueError("Unsupported JSON dataset format (expected list or {data: [...]})")

            items = []
            for obj in rows:
                if not isinstance(obj, dict):
                    continue
                image_path = str(obj.get("image_path") or obj.get("image") or "").strip()
                text = str(obj.get("text") or obj.get("caption") or "").strip()
                if not image_path:
                    continue
                img = Path(image_path)
                if not img.is_absolute():
                    img = (base_dir / img).resolve()
                items.append({"image_path": str(img), "text": text})
            if not items:
                raise ValueError(f"No samples found in json: {root}")
            return items

        raise ValueError(f"Unsupported dataset file type: {suffix}")

    # Directory mode
    meta_jsonl = root / "metadata.jsonl"
    if meta_jsonl.exists():
        return _load_pairs(str(meta_jsonl))

    images_dir = root / "images" if (root / "images").exists() else root
    captions_dir = root / "captions" if (root / "captions").exists() else images_dir

    images = sorted([p for p in images_dir.rglob("*") if p.is_file() and _is_image(p)])
    if not images:
        raise ValueError(f"No images found under: {images_dir}")

    items: List[Dict[str, str]] = []
    for img in images:
        caption_file = captions_dir / f"{img.stem}.txt"
        if not caption_file.exists() and captions_dir != img.parent:
            fallback = img.parent / f"{img.stem}.txt"
            if fallback.exists():
                caption_file = fallback
        items.append({"image_path": str(img), "text": _read_text(caption_file)})
    return items


class _ImageCaptionDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, str]], resolution: int) -> None:
        self.pairs = pairs
        self.resolution = int(resolution)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.pairs[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        arr = np.array(image).astype(np.float32) / 127.5 - 1.0
        pixel_values = torch.from_numpy(arr).permute(2, 0, 1)
        return {"pixel_values": pixel_values, "caption": item.get("text", "")}


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    captions = [str(b.get("caption") or "").strip() for b in batch]
    return {"pixel_values": pixel_values, "captions": captions}


def _encode_prompts_sdxl(
    captions: List[str],
    *,
    tokenizer: Any,
    tokenizer_2: Any,
    text_encoder: Any,
    text_encoder_2: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    text_inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs_2 = tokenizer_2(
        captions,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_inputs.input_ids.to(device)
    input_ids_2 = text_inputs_2.input_ids.to(device)

    with torch.no_grad():
        enc_out = text_encoder(input_ids, output_hidden_states=True)
        enc_out_2 = text_encoder_2(input_ids_2, output_hidden_states=True)

    prompt_embeds_1 = enc_out.hidden_states[-2]
    prompt_embeds_2 = enc_out_2.hidden_states[-2]

    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1).to(dtype)

    pooled = None
    if hasattr(enc_out_2, "text_embeds") and enc_out_2.text_embeds is not None:
        pooled = enc_out_2.text_embeds
    elif hasattr(enc_out_2, "pooler_output") and enc_out_2.pooler_output is not None:
        pooled = enc_out_2.pooler_output
    elif isinstance(enc_out_2, (tuple, list)) and len(enc_out_2) > 1:
        pooled = enc_out_2[1]
    if pooled is None:
        pooled = prompt_embeds_2.mean(dim=1)

    pooled_prompt_embeds = pooled.to(dtype)
    return prompt_embeds, pooled_prompt_embeds


def train_sdxl_lora(
    *,
    base_model: str,
    dataset_path: str,
    output_name: str,
    run_dir: Path,
    lora_out_dir: Path,
    resolution: int = 1024,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_train_steps: int = 1000,
    learning_rate: float = 1e-4,
    lora_rank: int = 16,
    seed: int = 42,
    mixed_precision: str = "fp16",
    save_steps: int = 500,
    progress_cb: Optional[ProgressCallback] = None,
) -> Path:
    """
    Minimal SDXL LoRA training loop (UNet attention processors).

    Notes:
    - Designed for single-GPU 16GB setups (use fp16 + grad checkpointing + 8bit optimizer).
    - Expects dataset_path to be a folder of images (+ optional captions/*.txt), or a jsonl/json file.
    """

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers.loaders import AttnProcsLayers
    from diffusers.models.attention_processor import LoRAAttnProcessor
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
        StableDiffusionXLPipeline,
    )
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    run_dir = Path(run_dir)
    lora_out_dir = Path(lora_out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lora_out_dir.mkdir(parents=True, exist_ok=True)

    # Persist config for reproducibility
    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "base_model": base_model,
                "dataset_path": dataset_path,
                "output_name": output_name,
                "resolution": resolution,
                "train_batch_size": train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "max_train_steps": max_train_steps,
                "learning_rate": learning_rate,
                "lora_rank": lora_rank,
                "seed": seed,
                "mixed_precision": mixed_precision,
                "save_steps": save_steps,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        mixed_precision=str(mixed_precision),
    )
    set_seed(int(seed))

    weight_dtype = torch.float32
    if str(mixed_precision).lower() == "fp16":
        weight_dtype = torch.float16
    elif str(mixed_precision).lower() == "bf16":
        weight_dtype = torch.bfloat16

    logger.info("[SDXL LoRA] Loading base model: %s", base_model)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=weight_dtype,
        use_safetensors=True,
        variant="fp16" if weight_dtype == torch.float16 else None,
    )

    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    vae = pipe.vae
    unet = pipe.unet

    # We don't need the pipeline object anymore
    del pipe

    # Freeze non-LoRA parameters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    try:
        unet.enable_gradient_checkpointing()
    except Exception:
        pass

    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # Inject LoRA processors into UNet attention blocks
    lora_attn_procs: Dict[str, Any] = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            continue

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=int(lora_rank),
        )

    unet.set_attn_processor(lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    # Optimizer (8-bit AdamW if available)
    optimizer_cls: Any = torch.optim.AdamW
    try:
        import bitsandbytes as bnb  # type: ignore

        optimizer_cls = bnb.optim.AdamW8bit
    except Exception:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(unet_lora_layers.parameters(), lr=float(learning_rate))

    # Scheduler (DDPM)
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    # Dataset / dataloader
    pairs = _load_pairs(dataset_path)
    dataset = _ImageCaptionDataset(pairs, resolution=int(resolution))
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
    )

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    unet.train()

    global_step = 0
    num_update_steps_per_epoch = math.ceil(len(dataloader) / int(gradient_accumulation_steps))
    num_train_epochs = math.ceil(int(max_train_steps) / max(1, num_update_steps_per_epoch))

    checkpoint_root = run_dir / "checkpoints"
    if accelerator.is_main_process:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_train_epochs):
        for _, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                captions = batch["captions"]

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                prompt_embeds, pooled_prompt_embeds = _encode_prompts_sdxl(
                    captions,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )

                # SDXL additional conditioning: (orig_h, orig_w, crop_top, crop_left, target_h, target_w)
                add_time_ids = torch.tensor(
                    [resolution, resolution, 0, 0, resolution, resolution],
                    device=accelerator.device,
                    dtype=weight_dtype,
                ).repeat(bsz, 1)

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
                    return_dict=False,
                )[0]

                target = noise
                if getattr(noise_scheduler.config, "prediction_type", "epsilon") == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if progress_cb:
                    progress_cb(global_step, int(max_train_steps), float(loss.detach().item()))

                if accelerator.is_main_process and int(save_steps) > 0 and global_step % int(save_steps) == 0:
                    ckpt_dir = checkpoint_root / f"step_{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(unet)
                    unwrapped.save_attn_procs(str(ckpt_dir))

            if global_step >= int(max_train_steps):
                break
        if global_step >= int(max_train_steps):
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(unet)
        unwrapped.save_attn_procs(str(lora_out_dir))

        info = {
            "name": output_name,
            "base_model": base_model,
            "model_type": "sdxl",
            "rank": int(lora_rank),
            "resolution": int(resolution),
            "max_train_steps": int(max_train_steps),
        }
        (lora_out_dir / "info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (lora_out_dir / "MODEL_CARD.md").write_text(
            f"# SDXL LoRA: {output_name}\n\n- Base: `{base_model}`\n- Rank: `{lora_rank}`\n- Steps: `{max_train_steps}`\n",
            encoding="utf-8",
        )

    accelerator.end_training()
    return lora_out_dir

