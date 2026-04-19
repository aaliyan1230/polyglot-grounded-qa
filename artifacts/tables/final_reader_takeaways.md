## 7) Reader takeaway summary

This section converts the latest delta table into plain-language findings.

- Highest delta_grounded_trust_score (including diagnostics): **oracle-upper-bound** ($\Delta$ = 0.8169).
- Recommended practical variant today (leaderboard gate-pass): **grounded-heuristic-v1** ($\Delta$ trust = 0.8169, $\Delta$ F1 = 0.9713, $\Delta$ abstain = 0.1132, $\Delta$ citation precision = 1.0000, $\Delta$ citation recall = 1.0000).
- Variants with simultaneous gains on grounding and answer quality: oracle-upper-bound, grounded-heuristic-v1, tuned-control-baseline, tuned-adapter-v1, base-model-prompted-v1.

### Where we stand
- The evaluation loop is reproducible end-to-end and tracks a composite grounded trust score.
- Practical ranking is artifact-backed via finetune_variant_leaderboard outputs.
- Next milestone is to add and compare a real trained-adapter row under this same trust-first rubric.

### Hybrid retrieval snapshot
- Strongest current KG coverage appears in **base** with path yield rate = 1.0000.
- Hybrid graph support delta vs text-only = 2.3825 for that slice.
- High leakage risk rate for that slice = 0.0000.

### Hybrid abstention snapshot
- Strongest abstention variant: **hybrid-graph-aware-abstain** with abstain accuracy = 0.9499.
- Abstain precision = 0.7361, abstain recall = 0.8689.
- Average graph support score for that variant = 2.3823.
