from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train grounded QA SFT adapter with Unsloth.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/finetune/cloud_unsloth_qlora.yaml"),
        help="Path to Unsloth finetuning config.",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/benchmarks/finetune/formatted/train.text.jsonl"),
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("data/benchmarks/finetune/formatted/val.text.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/finetune_unsloth"),
    )
    args = parser.parse_args()

    try:
        datasets_module = importlib.import_module("datasets")
        trl_module = importlib.import_module("trl")
        load_dataset = datasets_module.load_dataset
        SFTConfig = trl_module.SFTConfig
        SFTTrainer = trl_module.SFTTrainer
    except Exception as exc:  # pragma: no cover - runtime guidance only
        raise RuntimeError(
            "Missing required training dependencies. Install with:\n"
            "pip install datasets trl peft accelerate transformers bitsandbytes"
        ) from exc

    project_root = Path(__file__).resolve().parents[1]
    cfg = _load_yaml(project_root / args.config)

    model_cfg = cfg.get("model", {})
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})

    model_name = str(model_cfg.get("name", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"))
    max_seq_length = int(train_cfg.get("max_seq_length", 2048))

    use_unsloth = True
    try:
        unsloth_module = importlib.import_module("unsloth")
        FastLanguageModel = unsloth_module.FastLanguageModel
    except Exception:
        use_unsloth = False

    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lora_cfg.get("rank", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 16)),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            target_modules=list(
                lora_cfg.get(
                    "target_modules",
                    [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                )
            ),
            bias="none",
            use_gradient_checkpointing=str(lora_cfg.get("use_gradient_checkpointing", "unsloth")),
        )
        print("Using Unsloth training path.")
    else:
        print("Unsloth import failed; using transformers+peft fallback path.")
        transformers_module = importlib.import_module("transformers")
        peft_module = importlib.import_module("peft")
        torch_module = importlib.import_module("torch")

        AutoModelForCausalLM = transformers_module.AutoModelForCausalLM
        AutoTokenizer = transformers_module.AutoTokenizer
        BitsAndBytesConfig = getattr(transformers_module, "BitsAndBytesConfig", None)
        LoraConfig = peft_module.LoraConfig
        get_peft_model = peft_module.get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_in_4bit = bool(model_cfg.get("load_in_4bit", True))
        model_kwargs: dict[str, Any] = {"device_map": "auto"}
        if load_in_4bit and BitsAndBytesConfig is not None:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            model_kwargs["torch_dtype"] = getattr(torch_module, "float16")

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        peft_cfg = LoraConfig(
            r=int(lora_cfg.get("rank", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 16)),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            target_modules=list(
                lora_cfg.get(
                    "target_modules",
                    [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                )
            ),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

    train_ds = load_dataset(
        "json",
        data_files=str(project_root / args.train_file),
        split="train",
    )
    val_ds = load_dataset(
        "json",
        data_files=str(project_root / args.val_file),
        split="train",
    )

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        args=SFTConfig(
            output_dir=str(output_dir),
            max_seq_length=max_seq_length,
            per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 2)),
            gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
            learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
            warmup_steps=int(train_cfg.get("warmup_steps", 20)),
            max_steps=int(train_cfg.get("max_steps", 600)),
            logging_steps=int(train_cfg.get("logging_steps", 10)),
            save_steps=int(train_cfg.get("save_steps", 50)),
            optim=str(train_cfg.get("optim", "adamw_8bit")),
            seed=int(train_cfg.get("seed", 42)),
            eval_strategy="steps",
            eval_steps=int(train_cfg.get("save_steps", 50)),
        ),
    )

    trainer.train()
    model.save_pretrained(str(output_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(output_dir / "lora_adapter"))
    print(f"Saved adapter to {output_dir / 'lora_adapter'}")


if __name__ == "__main__":
    main()
