# Data Directory — Structure, Formats, and Processing Notes

This document describes the `data/` folder layout, naming conventions, required files, and the expected JSON/BibTeX artifacts produced by the scraping and processing pipeline. It is written to be concise, machine-friendly, and suitable for production-grade ingestion.

**Goals**
- **Consistency:** uniform folder and file naming across papers.
- **Reproducibility:** clear, minimal JSON schemas for downstream tasks.
- **Traceability:** preserve original TeX/BibTeX sources while providing normalized artifacts for ML workflows.

**Top-level layout**

- `data/raw/` — raw scraped outputs. Each subfolder corresponds to a single paper, named with arXiv-style `yymm-id` (e.g., `2412-15272`).
- `data/processed/` — cleaned + parsed outputs used by ML/analysis components. Each paper has a matching `yymm-id` folder under `processed/`.
- `data/README.md` — this file.

Example:

```
data/
  raw/
    2412-15272/
      tex/
        2412-15272v1/
          main.tex
          custom.tex
          custom.bib
          images/
            fig1.pdf
      metadata.json
      references.json
  processed/
    2412-15272/
      hierarchy.json
      refs.bib
      metadata.json
      references.json
      pred.json
```

Directory and file conventions

- Paper folder name: `yymm-id` (replace the dot in arXiv id with a dash, e.g., `2310-12345`).
- Versioned TeX subfolders: keep the original TeX source hierarchy. Use `...v<version>` for author-uploaded versions (e.g., `2412-15272v1`).
- Never alter `.tex` or `.bib` contents during scraping. Keep a byte-for-byte copy in `raw/tex/...`.

Contents of key files

- `tex/` (raw): contains the full TeX source tree as uploaded by the authors. This must preserve subfolder structure and file names (e.g., style files, images, nested tex files). BibTeX files authored by the paper (`*.bib`) must remain intact and unmodified.

- `metadata.json` (raw and processed): paper-level metadata. Minimal required fields:

```json
{
  "paper_title": "<title string>",
  "authors": ["Author A", "Author B"],
  "submission_date": "YYYY-MM-DD",     
  "revised_dates": ["YYYY-MM-DD", ...],
  "publication_venue": "<venue or empty>"
}
```

- `references.json` (raw and processed): dictionary mapping referenced `yymm-id` -> metadata. Only include references for which a matching arXiv id was found. Example entry:

```json
{
  "2308-11432": {
    "title": "Paper title",
    "authors": ["A. One", "B. Two"],
    "submission_date": "2023-08-15",
    "semantic_scholar_id": "28c6ac..."
  }
}
```

- `refs.bib` (processed): unified, deduplicated BibTeX collection for the publication after hierarchy construction. This file is suitable for downstream citation-matching and human inspection.

- `hierarchy.json` (processed): canonical, deduplicated representation of parsed textual elements across all versions. Key points:
  - Field `elements` contains mapping from element-id -> text content.
  - Element ids must encode source identity (arXiv id or SS id) and a local discriminator.
  - Exact full-text duplicates across versions collapse to a single element id.

- `pred.json` (processed, optional): predictions from reference-matching ML models (only present for train/val/test partitions). Minimal structure:

```json
{
  "partition": "train|valid|test",
  "groundtruth": {"bibkey1": "2412-15272", ...},
  "prediction": {"bibkey1": ["2412-15272","2310-12345", ...], ...}
}
```

Quality rules and deduplication

- Elements are deduplicated by exact full-text match — if two extracted fragments are byte-identical, keep one canonical id and reference it in `hierarchy.json`.
- When merging BibTeX entries into `refs.bib`, prefer author-uploaded `.bib` entries (raw `.bib` files). Use Semantic Scholar / external metadata only to augment missing fields, never to replace author-provided entries.

Processing notes and responsibilities

- Scraper responsibilities (produces `data/raw/<yymm-id>/`):
  - Download and save the entire TeX source tree without modifications.
  - Produce `metadata.json` and `references.json` for the raw crawl.

- Post-processing responsibilities (produces `data/processed/<yymm-id>/`):
  - Parse TeX sources into a hierarchical text representation and write `hierarchy.json`.
  - Deduplicate and unify BibTeX entries into `refs.bib`.
  - Normalize and validate `metadata.json` and `references.json`.
  - Optional: run reference-matching models and emit `pred.json` for labeled partitions.

Validation and minimal checks

- `metadata.json` must parse as JSON and include `paper_title`, `authors`, `submission_date`, and `revised_dates` (array, possibly empty).
- `references.json` keys must match pattern `^[0-9]{4}-[0-9]{5}$` for arXiv-style `yymm-id` keys.
- `refs.bib` must be valid UTF-8 and parsable by common BibTeX parsers.

Usage examples

- To list all processed papers:

```bash
ls data/processed | head
```

- To run downstream ingestion (example):

```bash
python src/scraper/file_processor.py --input data/processed/2412-15272 --output /tmp/ingest.json
```

Best practices

- Preserve originals: never modify files under `data/raw/` after scraping — re-run processing if fixes are needed.
- Keep processed artifacts deterministic: processing scripts should be idempotent and produce identical `hierarchy.json`/`refs.bib` when re-run on the same raw inputs.
- Record processing provenance (tool, version, timestamp) in `processed/<yymm-id>/metadata.json` when possible.

Contact & contribution

If you find issues with the data schema or need new fields for downstream tasks, open an issue in the repository or contact the data maintainers.

---
Generated for the project pipeline. See `src/scraper/README.md` for code-level processing details.
