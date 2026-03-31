# polyglot-grounded-qa

Notebook-first multilingual grounded QA research repo with a reusable core package.

## Project goals

- Keep pipeline logic language-agnostic and stable.
- Add new language/locale support using language packs and inheritance.
- Prioritize reproducible notebook experiments over deployment complexity.
- Avoid model-gateway lock-in and use direct provider/local adapters.

## Architecture

- `src/polyglot_grounded_qa/core`: pipeline orchestration + config loading.
- `src/polyglot_grounded_qa/components`: retriever/reranker/generator/verifier/abstain interfaces and baseline components.
- `src/polyglot_grounded_qa/langpacks`: language-pack contract + registry.
- `configs/languages`: `base`, `tr`, `es`, `es-MX`, `fr` with inheritance.
- `notebooks`: narrative experiments and results.

## Retrieval backend choice

Default strategy is local and notebook-friendly:

- Primary: FAISS + BM25 (no always-on service required).
- Optional: LanceDB adapter for larger local experiments.

No external vector DB is required for v1.

## Environment setup

```bash
uv sync --extra dev --extra notebooks
uv run pre-commit install
```

Optional extras:

```bash
uv sync --extra retrieval --extra evaluation --extra llm
```

Fine-tuning data + training helpers:

```bash
uv sync --extra finetune
```

## Quick start

Build a seed index artifact:

```bash
uv run python scripts/build_index.py
```

Run a single query:

```bash
uv run python scripts/run_pipeline.py "What is grounded QA?" --language base
```

Run sample eval:

```bash
uv run python scripts/run_eval.py
```

Optional generator backend selection (default stays deterministic baseline):

```bash
# Use local Ollama if available
PGQA_GENERATOR_BACKEND=ollama uv run python scripts/run_pipeline.py "What is grounded QA?"

# Use Gemini if GEMINI_API_KEY is present
PGQA_GENERATOR_BACKEND=google uv run python scripts/run_pipeline.py "What is grounded QA?"
```

To validate backend wiring without local services or API calls, run mocked selection tests:

```bash
uv run pytest tests/unit/test_generator_selection.py
```

Run ablation sample:

```bash
uv run python scripts/run_ablation.py
```

Build first-pass fine-tuning dataset and balanced splits:

```bash
uv run python scripts/build_sft_dataset.py
uv run python scripts/split_sft_dataset.py
```

Generated outputs:

- `data/benchmarks/finetune/sft_dataset.jsonl`
- `data/benchmarks/finetune/train.jsonl`
- `data/benchmarks/finetune/val.jsonl`
- `data/benchmarks/finetune/test.jsonl`

Run the full fine-tuning data workflow in one command:

```bash
# Internal-only data pipeline
uv run python scripts/run_finetune_data_pipeline.py

# Include public multilingual QA ingestion when datasets is installed
uv run --with datasets python scripts/run_finetune_data_pipeline.py --with-public
```

Ingest multilingual public QA data (XQuAD) and merge with internal SFT set:

```bash
uv run --with datasets python scripts/ingest_public_qa.py --max-per-language 150
uv run python scripts/merge_sft_datasets.py
uv run python scripts/split_sft_dataset.py --input data/benchmarks/finetune/sft_dataset_merged.jsonl
```

Analyze SFT dataset quality and export summary tables/report:

```bash
uv run python scripts/analyze_sft_dataset.py --input data/benchmarks/finetune/sft_dataset_merged.jsonl
```

Format train/val/test splits for direct training:

```bash
uv run python scripts/format_sft_for_training.py
```

Training-ready files:

- `data/benchmarks/finetune/formatted/train.chat.jsonl`
- `data/benchmarks/finetune/formatted/val.chat.jsonl`
- `data/benchmarks/finetune/formatted/test.chat.jsonl`
- `data/benchmarks/finetune/formatted/train.text.jsonl`
- `data/benchmarks/finetune/formatted/val.text.jsonl`
- `data/benchmarks/finetune/formatted/test.text.jsonl`

Quality outputs:

- `artifacts/tables/sft_dataset_quality_overall.parquet`
- `artifacts/tables/sft_dataset_quality_by_language.parquet`
- `artifacts/tables/sft_dataset_quality_by_source.parquet`
- `artifacts/tables/sft_dataset_quality_report.md`

Fine-tuning presets:

- `configs/finetune/local_mlx_lora.yaml`
- `configs/finetune/cloud_unsloth_qlora.yaml`

Run Unsloth SFT training (Kaggle/Colab free GPU target):

```bash
uv run python scripts/train_unsloth_sft.py \
	--config configs/finetune/cloud_unsloth_qlora.yaml \
	--train-file data/benchmarks/finetune/formatted/train.text.jsonl \
	--val-file data/benchmarks/finetune/formatted/val.text.jsonl
```

Evaluate baseline or tuned predictions against finetune test split:

```bash
# Baseline pipeline evaluation
uv run python scripts/run_finetune_eval.py --variant baseline-pipeline

# Tuned model evaluation (predictions jsonl must include: id, answer, citations, abstained)
uv run python scripts/run_finetune_eval.py \
	--variant tuned-adapter \
	--predictions artifacts/runs/tuned_predictions.jsonl

# Append to evaluation history for cross-run comparison
uv run python scripts/run_finetune_eval.py --variant baseline-pipeline --append
uv run python scripts/run_finetune_eval.py \
	--variant tuned-adapter \
	--predictions artifacts/runs/tuned_predictions.jsonl \
	--append
```

Normalize raw model outputs before evaluation (when predictions are not already in canonical shape):

```bash
uv run python scripts/normalize_tuned_predictions.py \
	--input artifacts/runs/raw_model_predictions.jsonl \
	--output artifacts/runs/tuned_predictions.jsonl
```

Generate raw predictions for evaluation:

```bash
# Control path using current baseline pipeline
uv run python scripts/generate_tuned_predictions.py \
	--mode baseline-pipeline \
	--output artifacts/runs/raw_model_predictions_control.jsonl

# Grounded heuristic policy variant (non-oracle, retrieval-driven)
uv run python scripts/generate_tuned_predictions.py \
	--mode grounded-heuristic \
	--output artifacts/runs/raw_model_predictions_heuristic.jsonl

# HF + adapter path (for trained LoRA adapters)
uv run python scripts/generate_tuned_predictions.py \
	--mode hf-adapter \
	--base-model Qwen/Qwen2.5-3B-Instruct \
	--adapter-path artifacts/runs/finetune_unsloth/lora_adapter \
	--output artifacts/runs/raw_model_predictions.jsonl
```

Finetune evaluation outputs:

- `artifacts/runs/finetune_eval_rows.parquet`
- `artifacts/tables/finetune_eval_summary.parquet`
- `artifacts/tables/finetune_eval_by_language.parquet`
- `artifacts/tables/final_finetune_eval_deltas.parquet`
- `artifacts/tables/meaningful_result_snapshot.md`

Execute notebooks in sequence:

```bash
uv run python scripts/run_notebook_batch.py --kernel python3
```

## Phases

1. Foundation and contracts.
2. Baseline retrieval + cited generation.
3. Verification and abstention trust layer.
4. Evaluation harness and calibration.
5. Language pack expansion and locale overrides.
6. Ablations and final narrative outputs.

## Notes

- This repo intentionally avoids LiteLLM.
- Use direct adapters for OpenAI/Anthropic/Google or local adapters (Ollama/MLX/vLLM).
