# polyglot-grounded-qa

Multilingual grounded QA research repo focused on one question: can we improve trustworthiness by making answers retrieval-backed, citation-aware, and willing to abstain when evidence is weak?

This repository answers that with a reusable package, notebook-backed experiments, and artifactized evaluations rather than a demo-first app.

## What this repo achieved

As of the latest artifact snapshot, the strongest practical variant is `grounded-heuristic-v1`, evaluated against `baseline-pipeline` with a trust-first rubric.

| Variant | Practical | Gate pass | Delta trust | Delta answer F1 | Delta citation precision | Delta citation recall |
|---|---:|---:|---:|---:|---:|---:|
| grounded-heuristic-v1 | yes | yes | 0.6865 | 0.2256 | 1.0000 | 1.0000 |
| oracle-upper-bound | no | no | 0.8343 | 0.9648 | 1.0000 | 1.0000 |
| tuned-adapter-v1 | yes | no | 0.1171 | -0.0352 | 0.2069 | 0.2069 |
| tuned-control-baseline | no | no | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Main takeaways:

- The best non-oracle system materially improves abstention behavior, citation faithfulness, and answer quality at the same time.
- The repo now has a reproducible end-to-end evaluation loop with leaderboard artifacts instead of one-off notebook claims.
- The current trained adapter path is promising but does not yet beat the heuristic policy on the practical promotion gate.

Primary evidence:

- `artifacts/tables/meaningful_result_snapshot.md`
- `artifacts/tables/final_reader_takeaways.md`
- `artifacts/tables/finetune_variant_leaderboard.md`

## Methodology

The project is deliberately organized around trust, not raw answer rate.

```mermaid
flowchart LR
	A[Multilingual corpora] --> B[Indexing and retrieval]
	B --> C[Cited answer generation]
	C --> D[Verification and abstention policy]
	D --> E[Trust-first evaluation]
	E --> F[Leaderboards and final artifacts]
	E --> G[SFT dataset construction]
	G --> H[LoRA or QLoRA adapter training]
	H --> E
```

The methodology has five parts:

1. Retrieval stays external to the model so knowledge freshness lives in the index, not model weights.
2. Generation is required to ground answers in retrieved chunks and emit explicit citations.
3. Evaluation prioritizes abstention and citation quality, not just answer overlap.
4. Language support is added through language packs and inheritance instead of duplicating pipeline logic.
5. Fine-tuning is treated as a policy-improvement step that must beat the existing grounded heuristic under the same evaluator.

The trust-first composite used in finetune evaluation is:

`grounded_trust_score = 0.2 * abstain_accuracy + 0.3 * citation_precision + 0.3 * citation_recall + 0.2 * answer_token_f1`

Practical variants only get promoted when they improve grounding metrics and avoid answer-quality regressions.

## Read the results first

If you only open a few files, start here:

- `artifacts/tables/meaningful_result_snapshot.md`: concise outcome summary for the latest meaningful run.
- `artifacts/tables/final_reader_takeaways.md`: plain-language interpretation of the final tables.
- `artifacts/tables/finetune_variant_leaderboard.md`: ranking of baseline, heuristic, control, and tuned variants.
- `docs/zero_cost_finetuning_playbook.md`: training strategy for free or low-cost compute.
- `notebooks/80_final_results.ipynb`: narrative walkthrough of final outputs.
- `notebooks/85_colab_adapter_training.ipynb`: GPU-oriented adapter training workflow.

## Repo shape

Core code lives in `src/polyglot_grounded_qa` and is split into a few stable layers:

- `core`: pipeline orchestration and config loading.
- `components`: retriever, reranker, generator, verifier, and abstention interfaces.
- `langpacks`: language-pack contracts and registry.
- `eval`: evaluation logic and trust-focused metrics.

Research artifacts live alongside the code:

- `notebooks`: the narrative path from ingestion to final results.
- `scripts`: reproducible entry points for index building, eval, ablations, data prep, and training.
- `artifacts/runs`: prediction files and run outputs.
- `artifacts/tables`: leaderboard, deltas, summaries, and quality reports.

## What is already reproducible

This repo is strongest when treated as an artifact-backed research pipeline. The shortest path to reproducing the current story is:

```bash
uv sync --extra dev --extra notebooks --extra finetune
uv run python scripts/run_final_results_pipeline.py
```

That refreshes the final evaluation, ablation outputs, reader-facing summaries, and artifact contract checks from the current repo state.

For notebook execution without manual cell running:

```bash
uv run python scripts/run_notebook_batch.py --kernel python3
```

## Local vs GPU-heavy work

Most of the repo is intentionally local and notebook-friendly:

- `notebooks/00` through `notebooks/80_final_results.ipynb`: local CPU is sufficient.
- `notebooks/85_colab_adapter_training.ipynb`: use Kaggle or Colab T4.
- `scripts/train_unsloth_sft.py` and `scripts/run_trained_adapter_eval.py`: best run on Kaggle or Colab when adapter dependencies or GPU memory are a constraint.

The default retrieval path is also local-first:

- Primary backend: FAISS plus BM25.
- Optional backend: LanceDB for larger local experiments.
- No always-on external vector database is required.

## Fine-tuning stance

This project does not treat fine-tuning as the main story. The current evidence says the most meaningful gains come first from better grounding policy and evaluation discipline.

Fine-tuning exists here to answer a narrower question: can a lightweight adapter outperform the best retrieval-grounded heuristic under the same trust gate?

So far:

- The data pipeline for SFT is in place.
- Free-compute training paths exist for MLX LoRA and Unsloth QLoRA.
- The tuned adapter path is measurable with the same evaluator.
- The tuned adapter has not yet cleared the practical promotion bar.

## Minimal commands

If you want only the essential entry points, use these:

```bash
# Build or refresh an index
uv run python scripts/build_index.py

# Run the baseline pipeline on one query
uv run python scripts/run_pipeline.py "What is grounded QA?" --language base

# Run evaluation
uv run python scripts/run_eval.py

# Run ablations
uv run python scripts/run_ablation.py

# Build SFT data
uv run python scripts/run_finetune_data_pipeline.py --no-public
```

Training presets:

- `configs/finetune/local_mlx_lora.yaml`
- `configs/finetune/cloud_unsloth_qlora.yaml`

## Current limitations

- The verifier is still not a mature final component.
- The best trained adapter is not yet the best practical system.
- There are no checked-in figure assets yet; the strongest evidence currently lives in markdown tables and parquet outputs.

## Why this repo is interesting

The contribution here is not just that multilingual grounded QA runs. It is that the repo makes the tradeoffs inspectable:

- how retrieval and citation behavior affect trust,
- how abstention changes practical system quality,
- how language packs let one pipeline stretch across locales,
- and where fine-tuning actually helps or fails.

That makes this repo more useful as a research scaffold than a one-shot benchmark dump or a thin notebook collection.
