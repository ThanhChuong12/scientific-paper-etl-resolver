import os
import re
import time
import json
import tarfile
import shutil
import gzip
import tempfile
import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import BASE_DIR, START_MONTH, START_ID, END_ID, MAX_WORKERS, BATCH_SIZE, S2_DELAY
from utils import log, arxiv_id_to_folder, format_yymm_id, get_total_size, track_memory_usage
from data_fetchers import get_metadata, discover_versions_via_abs, get_semantic_data, download_eprint
from file_processor import extract_archive, copy_tex_and_bib_keep_structure, remove_figure_files

# Process a single arXiv paper
def process_paper(arxiv_base_id: str):
    # Performance measurement begin
    start_time = time.time()
    # Memory tracking thread
    stop_event = threading.Event()
    mem_stats = {"max": 0, "avg": 0}
    mem_thread = threading.Thread(target=track_memory_usage, args=(stop_event, mem_stats))
    mem_thread.start()

    log(f"Start processing {arxiv_base_id}")
    folder_name = arxiv_id_to_folder(arxiv_base_id)
    paper_dir = BASE_DIR / folder_name
    tex_root = paper_dir / "tex"
    paper_dir.mkdir(parents=True, exist_ok=True)
    tex_root.mkdir(parents=True, exist_ok=True)

    # Retrieve metadata from arXiv
    metadata = get_metadata(arxiv_base_id)
    versions = discover_versions_via_abs(arxiv_base_id)
    log(f"Discovered versions for {arxiv_base_id}: {versions}")
    
    # Retrieve Semantic Scholar venue + references
    s2_venue, refs = get_semantic_data(arxiv_base_id)

    if s2_venue:
        metadata['publication_venue'] = s2_venue

    total_tex = 0
    total_bib = 0
    total_size_before_paper = 0
    total_size_after_paper = 0
    got_tex = False

    # Process each version individually
    for ver in versions:
        arxiv_with_ver = f"{arxiv_base_id}{ver}"
        dest_filename = f"{format_yymm_id(arxiv_with_ver)}{ver}.tar.gz"
        tmp_dir = Path(tempfile.mkdtemp(prefix="arxiv_dl_"))
        dest_path = tmp_dir / dest_filename
        downloaded = None

        version_folder = tex_root / f"{format_yymm_id(arxiv_with_ver)}{ver}"
        version_folder.mkdir(parents=True, exist_ok=True)

        downloaded = download_eprint(arxiv_with_ver, dest_path)
        if not downloaded:
            downloaded = download_eprint(arxiv_base_id, dest_path)

        if not downloaded:
            log(f"Failed to download source for {arxiv_with_ver}, skipping this version (thư mục rỗng được giữ lại).")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        extracted_dir = tmp_dir / "extracted"
        try:
            extracted_dir.mkdir(exist_ok=True)
            if extract_archive(downloaded, extracted_dir):
                size_before_version = get_total_size(extracted_dir)
                total_size_before_paper += size_before_version
                
                tex_c, bib_c = copy_tex_and_bib_keep_structure(extracted_dir, version_folder)
                total_tex += tex_c
                total_bib += bib_c
                
                size_after_version = get_total_size(version_folder)
                total_size_after_paper += size_after_version
                
                if tex_c > 0:
                    got_tex = True
                
                removed = remove_figure_files(version_folder)
                
                log(f"Version {ver}: copied {tex_c} .tex, {bib_c} .bib, "
                    f"size_before_filter = {size_before_version} bytes, size_after_filter = {size_after_version} bytes")
            else:
                log(f"[ERROR] Failed to extract {downloaded}")
        except Exception as e:
            log(f"Error extracting {downloaded}: {e}")
        finally:
            try:
                if downloaded.exists():
                    downloaded.unlink(missing_ok=True)
                shutil.rmtree(extracted_dir, ignore_errors=True)
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
    status = 'success' if got_tex else 'no_tex'

    # Save metadata and references
    try:
        metadata_path = paper_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        log(f"Saved metadata.json ({metadata_path})")
    except Exception as e:
        log(f"Error saving metadata.json: {e}")

    try:
        references_path = paper_dir / "references.json"
        with open(references_path, 'w', encoding='utf-8') as f:
            json.dump(refs, f, indent=4, ensure_ascii=False)
        log(f"Saved references.json ({references_path}) entries = {len(refs)}")
    except Exception as e:
        log(f"Error saving references.json: {e}")

    # Performance measurement end
    stop_event.set()
    mem_thread.join()
    end_time = time.time()
    duration = end_time - start_time
    output_size_mb = total_size_after_paper / (1024*1024)
    perf_path = BASE_DIR / "performance.csv"
    write_header = not perf_path.exists()

    try:
        with open(perf_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "arxiv_id", "start_time", "end_time", "duration_sec",
                    "size_before_mb", "size_after_mb", "output_size_mb",
                    "max_ram_mb", "avg_ram_mb", "status"
                ])
            writer.writerow([
                arxiv_base_id,
                datetime.fromtimestamp(start_time).isoformat(),
                datetime.fromtimestamp(end_time).isoformat(),
                round(duration, 3),
                round(total_size_before_paper / (1024*1024), 3),
                round(total_size_after_paper / (1024*1024), 3),
                round(output_size_mb, 3),
                round(mem_stats["max"], 3),
                round(mem_stats["avg"], 3),
                status
            ])
        log("Wrote performance.csv entry OK")
    except Exception as e:
        log(f"Error writing performance.csv: {e}")

    return {
        "arxiv_id": arxiv_base_id,
        "versions_found": versions,
        "tex_files": total_tex,
        "bib_files": total_bib,
        "references_count": len(refs),
        "total_size_before": total_size_before_paper,
        "total_size_after": total_size_after_paper,
        "status": status,
        "max_ram_mb": mem_stats["max"],
        "avg_ram_mb": mem_stats["avg"]
    }

#  Process a batch of papers in parallel.
def process_paper_batch(paper_batch: List[str]) -> List[Dict]:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_paper = {executor.submit(process_paper, paper_id): paper_id
                           for paper_id in paper_batch}

        for future in as_completed(future_to_paper):
            paper_id = future_to_paper[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                log(f"Error processing {paper_id}: {e}")
                results.append({
                    'arxiv_id': paper_id,
                    'status': 'failed',
                    'error': str(e)
                })
    return results

# Main loop
def run_scraper():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    all_ids = [f"{START_MONTH}.{i:05d}" for i in range(START_ID, END_ID + 1)]
    total_papers = len(all_ids)

    log(f" --- STARTING SCRAPING OF {total_papers} PAPERS ---")
    log(f"Configuration: {MAX_WORKERS} workers, batch size {BATCH_SIZE}")

    start_time = datetime.now()
    all_summaries = []

    # Process in batches
    for batch_num, i in enumerate(range(0, total_papers, BATCH_SIZE)):
        batch_ids = all_ids[i:i + BATCH_SIZE]
        batch_start = i + 1
        batch_end = min(i + BATCH_SIZE, total_papers)
        log(f"Processing batch {batch_num + 1}: papers {batch_start}-{batch_end}")

        batch_summaries = process_paper_batch(batch_ids)
        all_summaries.extend(batch_summaries)

        if batch_num < (total_papers // BATCH_SIZE) - 1:
            time.sleep(2)

    # Statistics
    successful = [s for s in all_summaries if s.get('status') == 'success']
    failed = total_papers - len(successful)
    
    avg_size_before = 0
    avg_size_after = 0
    if successful:
        avg_size_before = sum(s.get('total_size_before', 0) for s in successful) / len(successful)
        avg_size_after = sum(s.get('total_size_after', 0) for s in successful) / len(successful)
    
    avg_references_per_paper = sum(s.get('references_count', 0) for s in successful) / len(successful) if successful else 0
    ref_success_count = sum(1 for s in successful if s.get('references_count', 0) > 0)
    ref_success_rate = (ref_success_count / len(successful) * 100) if successful else 0
    total_time = (datetime.now() - start_time).total_seconds()
    ram_values = [p.get("max_ram_mb", 0) for p in all_summaries if "max_ram_mb" in p]
    total_memory_usage_mb = round(sum(ram_values), 3) if ram_values else 0
    peak_memory_usage_mb = round(max(ram_values), 3) if ram_values else 0

    report = {
        "performance_metrics": {
            "total_papers": total_papers,
            "successful_papers": len(successful),
            "failed_papers": failed,
            "success_rate": f"{(len(successful)/total_papers*100):.1f}%" if total_papers>0 else "0%",
            "total_processing_time_seconds": round(total_time, 2),
            "total_processing_time_minutes": round(total_time / 60, 2),
            "papers_per_second": round(total_papers / total_time, 3) if total_time>0 else 0,
            "papers_per_minute": round(total_papers / (total_time / 60), 1) if total_time>0 else 0,
            "total_memory_usage_mb": total_memory_usage_mb,
            "peak_memory_usage_mb": peak_memory_usage_mb,
            "avg_paper_size_before_bytes": round(avg_size_before, 2),
            "avg_paper_size_after_bytes": round(avg_size_after, 2),
            "avg_references_per_paper": round(avg_references_per_paper, 2),
            "reference_metadata_success_rate": f"{ref_success_rate:.1f}%",
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "configuration": {
                "max_workers": MAX_WORKERS,
                "batch_size": BATCH_SIZE,
                "s2_delay": S2_DELAY
            }
        },
        "papers": all_summaries
    }

    report_path = BASE_DIR / "performance_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log(f"--- SCRAPING COMPLETED ---")
    log(f"Successful papers: {len(successful)}/{total_papers} ({(len(successful)/total_papers*100) if total_papers>0 else 0:.1f}%)")
    log(f"Average paper size before removing figures: {avg_size_before:.1f} bytes")
    log(f"Average paper size after removing figures (tex/bib only): {avg_size_after:.1f} bytes")
    log(f"Average references per successful paper: {avg_references_per_paper:.2f}")
    log(f"Reference metadata success rate: {ref_success_rate:.1f}%")
    log(f"Total time: {total_time/60:.1f} minutes")
    log(f"Total memory usage by all papers: {total_memory_usage_mb} MB")
    log(f"Max ram used: {peak_memory_usage_mb} MB")
    log(f"Report saved: {report_path}")
