from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, Optional[float]], None]


def _format_chatml(messages: Any) -> Optional[str]:
    if not isinstance(messages, list):
        return None
    parts: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip()
        content = m.get("content")
        if not role or content is None:
            continue
        parts.append(f"<|im_start|>{role}\n{str(content).strip()}<|im_end|>")
    if not parts:
        return None
    return "\n".join(parts)


def _load_text_dataset(dataset_path: str) -> List[str]:
    path = Path(dataset_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"dataset_path not found: {path}")
    if not path.is_file():
        raise ValueError("LLM LoRA dataset_path must be a file (jsonl recommended)")

    suffix = path.suffix.lower()
    if suffix not in {".jsonl", ".json"}:
        raise ValueError("Supported LLM dataset formats: .jsonl / .json")

    texts: List[str] = []
    if suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                texts.append(obj)
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("text"):
                texts.append(str(obj["text"]))
                continue
            chatml = _format_chatml(obj.get("messages"))
            if chatml:
                texts.append(chatml)
                continue
            if obj.get("prompt") or obj.get("completion"):
                texts.append(f"{obj.get('prompt','')}{obj.get('completion','')}")
                continue

    else:  # .json
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("Unsupported JSON dataset format (expected list or {data: [...]})")
        for obj in rows:
            if isinstance(obj, str):
                texts.append(obj)
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("text"):
                texts.append(str(obj["text"]))
                continue
            chatml = _format_chatml(obj.get("messages"))
            if chatml:
                texts.append(chatml)
                continue
            if obj.get("prompt") or obj.get("completion"):
                texts.append(f"{obj.get('prompt','')}{obj.get('completion','')}")
                continue

    texts = [t.strip() for t in texts if str(t).strip()]
    if not texts:
        raise ValueError(f"No training texts found in: {path}")
    return texts


@dataclass
class LLMLoraTrainConfig:
    base_model: str
    dataset_path: str
    output_name: str
    max_length: int = 2048
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 500
    warmup_steps: int = 50
    lr_scheduler: str = "cosine"
    seed: int = 42
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    use_4bit: bool = True


def train_llm_lora(
    *,
    config: LLMLoraTrainConfig,
    run_dir: Path,
    lora_out_dir: Path,
    progress_cb: Optional[ProgressCallback] = None,
) -> Path:
    """
    QLoRA-friendly LLM LoRA trainer (Transformers Trainer).

    Dataset format: jsonl/json with {text: "..."} or {messages:[{role,content}...]}.
    """

    from datasets import Dataset  # type: ignore
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    run_dir = Path(run_dir)
    lora_out_dir = Path(lora_out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lora_out_dir.mkdir(parents=True, exist_ok=True)

    texts = _load_text_dataset(config.dataset_path)
    dataset = Dataset.from_dict({"text": texts})

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=int(config.max_length),
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    quant_cfg = None
    if bool(config.use_4bit):
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=quant_cfg,
    )

    if bool(config.use_4bit):
        model = prepare_model_for_kbit_training(model)

    target_modules = config.target_modules or [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_cfg = LoraConfig(
        r=int(config.lora_rank),
        lora_alpha=int(config.lora_alpha),
        lora_dropout=float(config.lora_dropout),
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    class _ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
            if not progress_cb:
                return
            try:
                loss = None
                if isinstance(logs, dict):
                    loss_val = logs.get("loss")
                    if loss_val is not None:
                        loss = float(loss_val)
                progress_cb(int(state.global_step or 0), int(state.max_steps or 0), loss)
            except Exception:
                return

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=int(config.batch_size),
        gradient_accumulation_steps=int(config.gradient_accumulation_steps),
        learning_rate=float(config.learning_rate),
        warmup_steps=int(config.warmup_steps),
        max_steps=int(config.max_steps),
        lr_scheduler_type=str(config.lr_scheduler),
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        report_to=None,
        remove_unused_columns=False,
        seed=int(config.seed),
        optim="paged_adamw_8bit" if bool(config.use_4bit) else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[_ProgressCallback()],
    )

    logger.info("[LLM LoRA] Training start: %s", config.output_name)
    trainer.train()

    # Save final LoRA adapter into models root (lora_out_dir)
    model.save_pretrained(str(lora_out_dir))
    tokenizer.save_pretrained(str(lora_out_dir))

    (lora_out_dir / "info.json").write_text(
        json.dumps(
            {
                "name": config.output_name,
                "base_model": config.base_model,
                "model_type": "llm_lora",
                "lora_rank": int(config.lora_rank),
                "lora_alpha": int(config.lora_alpha),
                "target_modules": target_modules,
                "max_steps": int(config.max_steps),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (lora_out_dir / "MODEL_CARD.md").write_text(
        f"# LLM LoRA: {config.output_name}\n\n- Base: `{config.base_model}`\n- Steps: `{config.max_steps}`\n",
        encoding="utf-8",
    )

    return lora_out_dir

