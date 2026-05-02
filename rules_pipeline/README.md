# Rules Pipeline

Standalone pipeline for extracting a structured representation of the FCF rules PDF.

## Flow

1. Parse the PDF and remove repeated headers, footers, and page counters.
2. Rebuild a chapter map from the table of contents.
3. Split the document into small semantic anchors.
4. Merge those anchors into larger LLM batches so a rule, its system, focus, and exceptions stay together.
5. Extract deterministic patterns with regex: discipline powers, focus blocks, prerequisites, metadata, and table-like blocks.
6. Optionally send each large batch to a structured LLM provider.
7. Consolidate all section JSON files into one graph.

## Environment

Add these values to `.env` when needed:

```env
RULES_PDF_PATH=./docs/FCF - Regles MET.pdf
RULES_OUTPUT_DIR=./rules_pipeline/output
STRUCTURED_LLM_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
STRUCTURED_LLM_TIMEOUT_SECONDS=180
STRUCTURED_LLM_MAX_OUTPUT_TOKENS=4000
STRUCTURED_LLM_TEMPERATURE=0.1
RULES_BATCH_TARGET_CHARS=18000
RULES_BATCH_MAX_CHARS=26000
```

If `RULES_PDF_PATH` is omitted, the first PDF found in `DOCS_PATH` is used.

## Commands

Prepare sections and regex-only entities:

```powershell
.\.venv\Scripts\python.exe -m rules_pipeline.main --skip-llm
```

Smoke-test LLM extraction on the first 5 sections:

```powershell
.\.venv\Scripts\python.exe -m rules_pipeline.main --max-sections 5
```

Regenerate LLM files:

```powershell
.\.venv\Scripts\python.exe -m rules_pipeline.main --force
```

## Outputs

- `rules_pipeline/output/01_sections.json`
- `rules_pipeline/output/01_llm_batches.json`
- `rules_pipeline/output/02_regex_entities.json`
- `rules_pipeline/output/llm_sections/*.json`
- `rules_pipeline/output/03_consolidated_graph.json`
- `rules_pipeline/output/manifest.json`
