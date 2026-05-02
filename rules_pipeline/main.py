from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rules_pipeline.config import load_pipeline_config
from rules_pipeline.consolidate import consolidate_graph
from rules_pipeline.llm_clients import build_structured_client
from rules_pipeline.pdf_semantic import group_sections_for_llm, parse_pdf_semantically
from rules_pipeline.regex_extractors import extract_deterministic_entities

logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract structured rules from the FCF PDF.")
    parser.add_argument("--skip-llm", action="store_true", help="Only parse PDF and regex entities.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing LLM files.")
    parser.add_argument("--max-sections", type=int, default=0, help="Limit processed sections for smoke tests.")
    parser.add_argument("--max-batches", type=int, default=0, help="Limit new LLM batches processed in this run.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    cfg = load_pipeline_config()
    logger.info("Source PDF: %s", cfg.pdf_path)
    logger.info("Output directory: %s", cfg.output_dir)

    parsed = parse_pdf_semantically(cfg.pdf_path)
    micro_sections = parsed.sections[: args.max_sections] if args.max_sections > 0 else parsed.sections
    if args.max_sections > 0:
        parsed.sections = micro_sections

    _write_json(cfg.output_dir / "01_sections.json", parsed.to_dict())
    logger.info("Semantic micro-sections: %d", len(micro_sections))

    batches = group_sections_for_llm(
        micro_sections,
        target_chars=cfg.batch_target_chars,
        max_chars=cfg.batch_max_chars,
    )
    _write_json(
        cfg.output_dir / "01_llm_batches.json",
        {
            "target_chars": cfg.batch_target_chars,
            "max_chars": cfg.batch_max_chars,
            "batches": [batch.to_dict() for batch in batches],
        },
    )
    logger.info("LLM batches: %d", len(batches))

    regex_results = [extract_deterministic_entities(section) for section in micro_sections]
    _write_json(cfg.output_dir / "02_regex_entities.json", regex_results)

    llm_results: list[dict] = []
    llm_dir = cfg.output_dir / "llm_sections"
    if args.skip_llm:
        logger.info("Skipping LLM extraction.")
    else:
        client = build_structured_client(cfg)
        batch_regex_results = [extract_deterministic_entities(batch) for batch in batches]
        new_batches_done = 0
        for index, section in enumerate(batches, start=1):
            target = llm_dir / f"{section.section_id}.json"
            error_target = cfg.output_dir / "llm_errors" / f"{section.section_id}.json"
            if target.exists() and not args.force:
                cached = _read_json(target)
                if cached is not None:
                    llm_results.append(cached)
                    continue
            if args.max_batches > 0 and new_batches_done >= args.max_batches:
                logger.info("Stopping after %d new LLM batch(es).", new_batches_done)
                continue
            logger.info("[%d/%d] LLM extraction: %s", index, len(batches), section.section_id)
            try:
                result = client.extract(section, batch_regex_results[index - 1])
            except Exception as exc:
                logger.exception("LLM extraction failed for %s", section.section_id)
                _write_json(
                    error_target,
                    {
                        "section_id": section.section_id,
                        "index": index,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )
                continue
            _write_json(target, result)
            _write_json(
                cfg.output_dir / "progress.json",
                {
                    "last_completed_batch": section.section_id,
                    "last_completed_index": index,
                    "total_batches": len(batches),
                    "new_batches_done": new_batches_done + 1,
                },
            )
            llm_results.append(result)
            new_batches_done += 1

    graph = consolidate_graph(parsed, regex_results, llm_results)
    _write_json(cfg.output_dir / "03_consolidated_graph.json", graph)
    _write_json(
        cfg.output_dir / "manifest.json",
        {
            "pdf_path": str(cfg.pdf_path),
            "provider": cfg.provider,
            "skip_llm": args.skip_llm,
            "section_count": len(micro_sections),
            "batch_count": len(batches),
            "llm_result_count": len(llm_results),
            "max_batches": args.max_batches,
            "output_dir": str(cfg.output_dir),
        },
    )
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
