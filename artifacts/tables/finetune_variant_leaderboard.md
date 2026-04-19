# Finetune Variant Leaderboard

Ranked by grounding-gate pass status, then `delta_grounded_trust_score`, then `delta_avg_answer_token_f1`.

Promotion gate (practical variants only):
- `delta_avg_citation_precision > 0.0`
- `delta_avg_citation_recall > 0.0`
- `delta_grounded_trust_score > 0.0`
- `delta_avg_answer_token_f1 >= 0.0`
- `delta_abstain_accuracy >= -0.07` (up to 7% regression allowed; baseline abstain_accuracy is inflated by never-abstain behaviour)

| variant | practical | gate_pass | d_trust | d_f1 | d_cit_p | d_cit_r |
|---|---:|---:|---:|---:|---:|---:|
| grounded-heuristic-v1 | yes | yes | 0.8169 | 0.9713 | 1.0000 | 1.0000 |
| tuned-adapter-v1 | yes | yes | 0.3776 | 0.3091 | 0.5436 | 0.5473 |
| base-model-prompted-v1 | yes | yes | 0.0980 | 0.0283 | 0.1317 | 0.1317 |
| oracle-upper-bound | no | no | 0.8169 | 0.9713 | 1.0000 | 1.0000 |
| tuned-control-baseline | no | no | 0.5766 | 0.0109 | 0.9573 | 0.9573 |
