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

Run ablation sample:

```bash
uv run python scripts/run_ablation.py
```

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
