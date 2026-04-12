# SFT Dataset Manifest

This file defines the canonical grounded QA SFT dataset lineage and the scripts that generate each artifact.

## Canonical files

- `sft_dataset.jsonl`: Internal grounded QA SFT seed data.
- `public_sft_dataset.jsonl`: Optional public QA ingest (XQuAD-based).
- `sft_dataset_merged.jsonl`: Public + internal merged dataset.
- `train.jsonl`, `val.jsonl`, `test.jsonl`: Language-aware splits with abstention ratio checks.
- `formatted/*.text.jsonl`: Training-ready text format for Unsloth/TRL.
- `formatted/*.chat.jsonl`: Training-ready chat format (optional).

## Script lineage

1. `scripts/build_sft_dataset.py` -> `sft_dataset.jsonl`
2. `scripts/ingest_public_qa.py` -> `public_sft_dataset.jsonl` (optional)
3. `scripts/merge_sft_datasets.py` -> `sft_dataset_merged.jsonl`
4. `scripts/split_sft_dataset.py` -> `train.jsonl`, `val.jsonl`, `test.jsonl`
5. `scripts/format_sft_for_training.py` -> `formatted/*.text.jsonl`, `formatted/*.chat.jsonl`
6. `scripts/analyze_sft_dataset.py` -> quality tables and report

## Orchestration

Run the full pipeline via `scripts/run_finetune_data_pipeline.py`.

## Training defaults

The Unsloth training path expects `formatted/train.text.jsonl` and `formatted/val.text.jsonl` by default.
