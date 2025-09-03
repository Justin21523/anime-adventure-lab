# scripts/finetune_lora.py
"""
LoRA Fine-tuning Script
Minimal LoRA training setup with shared cache integration
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Shared cache bootstrap
import torch

AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/mnt/ai_warehouse/cache")
for k, v in {
    "HF_HOME": f"{AI_CACHE_ROOT}/hf",
    "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
    "HF_DATASETS_CACHE": f"{AI_CACHE_ROOT}/hf/datasets",
    "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
    "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import yaml

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Minimal LoRA trainer for LLM fine-tuning"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None

        # Setup output directory
        self.output_dir = (
            Path(AI_CACHE_ROOT)
            / "models"
            / "lora"
            / self.config["model"]["output_name"]
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_model(self) -> None:
        """Load base model and setup LoRA"""
        model_name = self.config["model"]["base_model"]

        logger.info(f"Loading base model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=f"{AI_CACHE_ROOT}/hf"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir=f"{AI_CACHE_ROOT}/hf",
        )

        # Setup LoRA configuration
        lora_config = LoraConfig(
            r=self.config["lora"]["rank"],
            lora_alpha=self.config["lora"]["alpha"],
            target_modules=self.config["lora"]["target_modules"],
            lora_dropout=self.config["lora"]["dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        logger.info("LoRA model setup completed")

    def prepare_dataset(self, data_path: str) -> Dataset:
        """Prepare training dataset"""
        logger.info(f"Loading dataset from: {data_path}")

        # Load data (assuming JSON lines format)
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        # Tokenize dataset
        def tokenize_function(examples):
            # Assuming conversation format
            inputs = []
            for item in examples:
                if isinstance(item, dict) and "text" in item:
                    inputs.append(item["text"])
                else:
                    inputs.append(str(item))

            # Tokenize
            tokenized = self.tokenizer(  # type: ignore
                inputs,
                truncation=True,
                padding=False,
                max_length=self.config["data"]["max_length"],
                return_tensors=None,
            )

            # For causal LM, labels are same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        dataset = Dataset.from_list(data)
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=dataset.column_names
        )

        return tokenized_dataset

    def train(self, train_data_path: str, eval_data_path: Optional[str] = None) -> None:
        """Run LoRA training"""

        # Load model
        self.load_model()

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_data_path)
        eval_dataset = None
        if eval_data_path:
            eval_dataset = self.prepare_dataset(eval_data_path)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config["training"]["epochs"],
            per_device_train_batch_size=self.config["training"]["batch_size"],
            gradient_accumulation_steps=self.config["training"][
                "gradient_accumulation_steps"
            ],
            learning_rate=self.config["training"]["learning_rate"],
            lr_scheduler_type=self.config["training"]["lr_scheduler"],
            warmup_steps=self.config["training"]["warmup_steps"],
            # Memory optimization
            fp16=True,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            # Logging and saving
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="steps" if eval_dataset else "no",  # type: ignore
            eval_steps=500 if eval_dataset else None,
            save_total_limit=3,
            # Misc
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard for simplicity
            seed=42,
        )

        # Import trainer (lazy import to avoid loading if not needed)
        from transformers import Trainer, DataCollatorForLanguageModeling

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False  # type: ignore Causal LM
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,  # type: ignore
        )

        # Start training
        logger.info("Starting LoRA training...")
        trainer.train()

        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)  # type: ignore

        # Save training metadata
        self._save_training_metadata()

        logger.info(f"Training completed. Model saved to: {self.output_dir}")

    def _save_training_metadata(self) -> None:
        """Save training configuration and metadata"""
        metadata = {
            "model_info": {
                "base_model": self.config["model"]["base_model"],
                "output_name": self.config["model"]["output_name"],
                "lora_rank": self.config["lora"]["rank"],
                "lora_alpha": self.config["lora"]["alpha"],
                "target_modules": self.config["lora"]["target_modules"],
            },
            "training_info": {
                **self.config["training"],
                "completed_at": datetime.now().isoformat(),
                "cache_root": AI_CACHE_ROOT,
            },
            "usage": "Text generation and chat completion",
            "limitations": "Fine-tuned for specific domain, may not generalize well",
            "license": "Same as base model license",
        }

        with open(
            self.output_dir / "training_metadata.json", "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Create MODEL_CARD.md
        model_card = f"""# LoRA Model Card: {self.config['model']['output_name']}

## Model Information
- **Base Model**: {self.config['model']['base_model']}
- **LoRA Rank**: {self.config['lora']['rank']}
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}

## Training Configuration
- **Epochs**: {self.config['training']['epochs']}
- **Learning Rate**: {self.config['training']['learning_rate']}
- **Batch Size**: {self.config['training']['batch_size']}

## Usage
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("{self.config['model']['base_model']}")
model = PeftModel.from_pretrained(base_model, "{self.output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{self.output_dir}")
```

## Limitations
- Fine-tuned for specific use cases
- Performance may vary on out-of-domain data
- Requires base model for inference

## License
Same as base model license. Check original model documentation.
"""

        with open(self.output_dir / "MODEL_CARD.md", "w", encoding="utf-8") as f:
            f.write(model_card)


def create_default_config(output_path: str) -> None:
    """Create default LoRA training configuration"""
    default_config = {
        "model": {
            "base_model": "Qwen/Qwen-7B-Chat",
            "output_name": "qwen7b_chat_lora_v1",
        },
        "lora": {
            "rank": 16,
            "alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "dropout": 0.1,
        },
        "training": {
            "epochs": 3,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "lr_scheduler": "cosine",
            "warmup_steps": 100,
        },
        "data": {
            "max_length": 2048,
            "train_file": "data/train.jsonl",
            "eval_file": "data/eval.jsonl",
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

    print(f"Default LoRA config created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for LLMs")
    parser.add_argument(
        "--config", required=True, help="Training configuration YAML file"
    )
    parser.add_argument(
        "--train-data", required=True, help="Training data file (JSONL)"
    )
    parser.add_argument("--eval-data", help="Evaluation data file (JSONL)")
    parser.add_argument("--create-config", help="Create default config file and exit")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create default config if requested
    if args.create_config:
        create_default_config(args.create_config)
        return

    # Validate inputs
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    if not Path(args.train_data).exists():
        raise FileNotFoundError(f"Training data not found: {args.train_data}")

    # Run training
    trainer = LoRATrainer(args.config)
    trainer.train(args.train_data, args.eval_data)

    print(
        f"""
🎉 LoRA training completed!

Model saved to: {trainer.output_dir}

To use the trained model:
1. Load in Python code (see MODEL_CARD.md)
2. Or use with the API server

Next steps:
- Test the model with sample inputs
- Evaluate on validation set
- Deploy to production if satisfied
"""
    )


if __name__ == "__main__":
    main()
