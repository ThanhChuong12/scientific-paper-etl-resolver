import re
import time
import json
from typing import List, Dict, Tuple, Optional
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from pathlib import Path

from http_client import session, enforce_rate_limit
from config import ARXIV_EPRINT_HEADERS, SEMANTIC_SCHOLAR_HEADERS
from utils import log
import requests

# Version discovery
def discover_versions_via_abs(arxiv_base_id: str, timeout: int = 20) -> List[str]:
    url = f"https://arxiv.org/abs/{arxiv_base_id}"
    try:
        enforce_rate_limit()
        r = session.get(url, headers=ARXIV_EPRINT_HEADERS, timeout=timeout)
        if r.status_code != 200:
            log(f"Warning: abs page {url} returned {r.status_code}; defaulting to v1")
            return ["v1"]
        html = r.text
        match = re.search(r"Submission history(.*?)</div>", html, re.S | re.I)
        history_block = match.group(1) if match else html
        versions = re.findall(r"\[v(\d+)\]", history_block)
        if not versions:
            return ["v1"]
        return [f"v{v}" for v in sorted(map(int, versions))]
    except Exception as e:
        log(f"Error discovering versions for {arxiv_base_id}: {e}. Defaulting to v1.")
        return ["v1"]

# Download source
def download_eprint(arxiv_id_with_version: str, dest_path: Path, timeout: int = 60) -> Optional[Path]:
    url = f"https://arxiv.org/e-print/{arxiv_id_with_version}"
    try:
        enforce_rate_limit()
        response = session.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            stream=True,
            timeout=timeout
        )
        if response.status_code in (404, 500, 503):
            log(f"[WARN] e-print unavailable for {arxiv_id_with_version} (HTTP {response.status_code})")
            return None
        content_type = response.headers.get("Content-Type", "").lower()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as file_handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_handle.write(chunk)
        size_kb = dest_path.stat().st_size / 1024.0
        log(f"[OK] Downloaded {arxiv_id_with_version} ({size_kb:.1f} KB) type={content_type}")
        return dest_path
    except Exception as e:
        log(f"[ERROR] Download failed for {arxiv_id_with_version}: {e}")
        return None

# Semantic Scholar Data: retrieve venue and reference metadata
def get_semantic_data(arxiv_base_id: str, max_retries: int = 5) -> Tuple[str, Dict[str, dict]]:
    api = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_base_id}"
    params = {
        "fields": ("venue,"
            "references.title,"
            "references.authors,"
            "references.externalIds,"
            "references.year,"
            "references.paperId")
    }
    
    attempts = 0
    while attempts < max_retries:
        try:
            attempts += 1
            enforce_rate_limit() # Control request frequency (thread-safe)
            
            headers = SEMANTIC_SCHOLAR_HEADERS.copy()
            if headers.get("x-api-key") is None or headers.get("x-api-key") == "":
                headers.pop("x-api-key", None)
                
            r = session.get(api, params=params, headers=headers, timeout=25)

            if r.status_code == 200:
                log(f"Semantic Scholar (200) OK for {arxiv_base_id}")
                break
            elif r.status_code == 429:
                log(f"Semantic Scholar rate limited (429) for {arxiv_base_id}. Attempt {attempts}/{max_retries}. Sleeping 10s...")
                time.sleep(10)
                continue
            elif r.status_code == 404:
                log(f"Semantic Scholar (404) Not Found for {arxiv_base_id}. Skipping references & venue.")
                return "", {}
            else:
                log(f"Semantic Scholar error {r.status_code} for {arxiv_base_id}. Attempt {attempts}/{max_retries}. Retrying in 5s...")
                time.sleep(5)
                continue
        except requests.RequestException as e:
            log(f"Semantic Scholar request exception for {arxiv_base_id}: {e}. Attempt {attempts}/{max_retries}. Retrying in 15s...")
            time.sleep(15)
            continue
            
    if attempts >= max_retries and ('r' not in locals() or r.status_code != 200):
        log(f"[WARN] Semantic Scholar failed after {max_retries} attempts for {arxiv_base_id}. Giving up.")
        return "", {}

    try:
        j = r.json() or {}
        venue = j.get('venue') or ""
        if venue:
            log(f"[OK] Retrieved venue for {arxiv_base_id}")

        refs = j.get('references') or []
        out = {}
        for ref in refs:
            if not isinstance(ref, dict): continue
            external = ref.get('externalIds') or {}
            ref_arx = external.get('ArXiv') or external.get('arXiv')
            if not ref_arx: continue
            base = ref_arx.split('v')[0]
            if '.' in base:
                parts = base.split('.')
                yymm_id = f"{parts[0]}-{parts[1]}"
            else:
                yymm_id = base.replace('.', '-')
            authors = [a.get('name') or "" for a in ref.get('authors', []) if isinstance(a, dict) and a.get('name')]
            year = ref.get('year')
            submission_date = f"{year}-01-01" if year else ""
            
            out[yymm_id] = {
                "paper_title": ref.get('title') or "",
                "authors": authors,
                "submission_date": submission_date,
                "semantic_scholar_id": ref.get('paperId') or ""
            }
        log(f"[OK] Retrieved {len(out)} references for {arxiv_base_id}")
        return venue, out
    except Exception as e:
        log(f"Error parsing Semantic Scholar JSON for {arxiv_base_id}: {e}")
        return "", {}

# Metadata extraction: Scrape metadata from arXiv /abs page
def get_metadata(arxiv_base_id: str):
    url = f"https://arxiv.org/abs/{arxiv_base_id}"
    try:
        enforce_rate_limit()
        r = session.get(url, headers=ARXIV_EPRINT_HEADERS, timeout=20)
        if r.status_code != 200:
            log(f"abs page error {r.status_code} for {arxiv_base_id}")
            return {
                "paper_title": "", "authors": [], "submission_date": "",
                "revised_dates": [], "publication_venue": ""
            }

        soup = BeautifulSoup(r.text, 'html.parser')

        # Title
        title_tag = soup.find('h1', class_='title')
        title = title_tag.get_text(strip=True).replace('Title:', '').strip() if title_tag else ""

        # Authors
        authors_tag = soup.find('div', class_='authors')
        authors = []
        if authors_tag:
            links = authors_tag.find_all('a')
            if links:
                authors = [a.text.strip() for a in links if a.text.strip()]
            else:
                text = authors_tag.get_text(separator=',').replace('Authors:', '')
                authors = [s.strip() for s in text.split(',') if s.strip()]

        # Submission history
        revised_dates = []
        submission_block = None
        hist_div = soup.find('div', class_='submission-history')
        if hist_div:
            submission_block = hist_div.get_text("\n", strip=True)
        else:
            m = re.search(r'Submission history(.*)', r.text, re.S | re.I)
            if m:
                submission_block = m.group(1)

        if submission_block:
            pattern = re.compile(
                r'\[v(\d+)\]\s+([A-Za-z]{3},\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4})'
            )
            versions = pattern.findall(submission_block)
            versions = sorted(versions, key=lambda x: int(x[0]))
            for vnum, date_str in versions:
                try:
                    dt = datetime.strptime(date_str.strip(), "%a, %d %b %Y")
                except ValueError:
                    try:
                        dt = datetime.strptime(date_str.strip(), "%d %b %Y")
                    except Exception:
                        continue
                parsed = dt.strftime("%Y-%m-%d")
                if parsed not in revised_dates:
                    revised_dates.append(parsed)
        submission_date = revised_dates[0] if revised_dates else ""
        
        publication_venue = ""
        cat_tag = soup.find('span', class_='primary-subject')
        if cat_tag:
            publication_venue = cat_tag.get_text(strip=True)

        return {
            "paper_title": title,
            "authors": authors,
            "submission_date": submission_date,
            "revised_dates": revised_dates,
            "publication_venue": publication_venue
        }
    except Exception as e:
        log(f"Error parsing abs for metadata {arxiv_base_id}: {e}")
        return {
            "paper_title": "", "authors": [], "submission_date": "",
            "revised_dates": [], "publication_venue": ""
        }