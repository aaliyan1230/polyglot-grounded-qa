# Meaningful Result Snapshot

Date: 2026-03-31

Evaluation source:
- `scripts/run_finetune_eval.py`
- `artifacts/tables/final_finetune_eval_deltas.parquet`

Baseline reference:
- Variant: `baseline-pipeline`

New non-oracle variant:
- Variant: `grounded-heuristic-v1`

Observed deltas vs baseline:
- `delta_abstain_accuracy`: `+0.206897`
- `delta_avg_citation_precision`: `+1.000000`
- `delta_avg_citation_recall`: `+1.000000`
- `delta_avg_answer_token_f1`: `+0.225564`

Interpretation:
- This variant materially improves the repo's north-star trust metrics (abstention and citation faithfulness) without using oracle labels at inference time.
- It is already a meaningful repo result even without any fine-tuning gains.
- It establishes a strong practical target that future adapter training must beat rather than the main claim the repo depends on.

Control sanity check:
- Variant `tuned-control-baseline` produces zero deltas, confirming the evaluation path is not inflating gains.

Current framing:
- The strongest practical story in this repo is the grounded heuristic plus the trust-first evaluator.
- Fine-tuning remains an optional optimization track and is only interesting when it clears that existing bar.
