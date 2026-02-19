import os
import re
import shutil
import gzip
import tarfile
from pathlib import Path
from typing import Tuple

from utils import log
from config import IMAGE_EXTS

# Binary detection
def is_binary_file(content: bytes) -> bool:
    if len(content) < 4: return False
    binary_sigs = [
        b"\x25\x50\x44\x46",  # PDF
        b"\xFF\xD8\xFF",      # JPEG
        b"\x89\x50\x4E\x47",  # PNG
        b"\x47\x49\x46\x38",  # GIF
        b"\x42\x4D",          # BMP
        b"\x1F\x8B\x08",      # GZIP
        b"\x50\x4B\x03\x04",  # ZIP
        b'\x52\x61\x72\x21',  # RAR
    ]
    if any(content.startswith(sig) for sig in binary_sigs):
        return True
    printable_count = sum(
        1 for b in content if 32 <= b <= 126 or b in [9, 10, 13]
    )
    printable_ratio = printable_count / len(content) if len(content) > 0 else 0
    return printable_ratio < 0.7

# Determine whether a file is likely a LaTeX source file
def is_latex_file(file_path: Path) -> bool:
    try:
        if not file_path.exists() or file_path.stat().st_size == 0:
            return False
        with open(file_path, 'rb') as f:
            content_start = f.read(4096)
        if is_binary_file(content_start):
            return False
        try:
            text = content_start.decode('utf-8', errors='ignore')
        except:
            text = content_start.decode('latin-1', errors='ignore')
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)
        latex_patterns = [
            r'\\documentclass', r'\\begin{document}', r'\\section{',
            r'\\usepackage', r'\\newcommand', r'\\input{'
        ]
        pattern_count = 0
        for pattern in latex_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_count += 1
        if pattern_count >= 2: return True
        if "\\documentclass" in text.lower() and "\\begin{document}" in text.lower():
            return True
        return False
    except Exception as e:
        log(f"[DEBUG] Error checking if {file_path} is LaTeX: {e}")
        return False

# Recursive extraction
def extract_recursive(archive_path: Path, extract_dir: Path, depth: int = 0) -> bool:
    if depth > 3:
        log(f"[WARN] Maximum recursion depth reached for {archive_path}")
        return False
    try:
        if not archive_path.exists():
            log(f"[WARN] Archive not found: {archive_path}")
            return False
        extract_dir.mkdir(parents=True, exist_ok=True)
        # CASE 1: TAR archive
        try:
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=extract_dir)
            log(f"[INFO] Extracted TAR archive: {archive_path.name}")
            for item in extract_dir.iterdir():
                if item.is_file() and item.suffix in [".tar", ".gz", ".tar.gz"]:
                    nested_dir = extract_dir / f"nested_extract_{depth}"
                    if extract_recursive(item, nested_dir, depth + 1):
                        for inner in nested_dir.iterdir():
                            target = extract_dir / inner.name
                            if target.exists():
                                target = extract_dir / f"{nested_dir.name}_{inner.name}"
                            shutil.move(str(inner), str(target))
                        shutil.rmtree(nested_dir)
                    item.unlink(missing_ok=True)
            return True
        except (tarfile.ReadError, tarfile.CompressionError):
            pass
        # CASE 2: GZIP single-file
        try:
            with gzip.open(archive_path, "rb") as f_in:
                decompressed_data = f_in.read()
            output_name = (
                archive_path.stem if archive_path.suffix == ".gz"
                else archive_path.name
            )
            output_path = extract_dir / output_name
            with open(output_path, "wb") as f_out:
                f_out.write(decompressed_data)
            log(f"[INFO] Decompressed GZIP file: {output_name}")
            if output_path.suffix == ".tar":
                try:
                    if extract_recursive(output_path, extract_dir, depth + 1):
                        output_path.unlink(missing_ok=True)
                        return True
                except Exception as e:
                    log(f"[WARN] Nested TAR extraction failed: {e}")
            try:
                if is_latex_file(output_path):
                    tex_path = output_path.with_suffix(".tex")
                    try:
                        output_path.rename(tex_path)
                    except Exception:
                        shutil.copy2(output_path, tex_path)
                    log(f"[INFO] LaTeX detected: renamed/copied to {tex_path.name}")
            except Exception as e:
                log(f"[DEBUG] Error during LaTeX detection: {e}")
            return True
        except Exception:
            return False
    except Exception as e:
        log(f"[ERROR] extract_recursive failed: {e}")
        return False

# Extraction Wrapper
def extract_archive(archive_path: Path, extract_dir: Path) -> bool:
    try:
        if not archive_path.exists() or archive_path.stat().st_size == 0:
            log(f"[WARN] Archive missing or empty: {archive_path}")
            return False
        extract_dir.mkdir(parents=True, exist_ok=True)
        ok = extract_recursive(archive_path, extract_dir)
        if not ok:
            log(f"[ERROR] Extraction failed for {archive_path}")
            return False
        
        all_files = list(extract_dir.rglob("*"))
        tex_files = [f for f in all_files if f.is_file() and f.suffix.lower() == ".tex"]
        bib_files = [f for f in all_files if f.is_file() and f.suffix.lower() == ".bib"]
        log(f"[EXTRACTION RESULT] Total files: {len(all_files)}, .tex: {len(tex_files)}, .bib: {len(bib_files)}")
        
        return len(tex_files) > 0 or len(bib_files) > 0
    except Exception as e:
        log(f"[ERROR] Extraction wrapper failed: {e}")
        return False

def remove_figure_files(path_root: Path) -> int:
    removed = 0
    for root, dirs, files in os.walk(path_root):
        for fname in files:
            if Path(fname).suffix.lower() in IMAGE_EXTS:
                try:
                    os.remove(Path(root) / fname)
                    removed += 1
                except Exception:
                    pass
    return removed

# Copy all .tex and .bib files from the extracted source
def copy_tex_and_bib_keep_structure(extracted_root: Path, target_version_dir: Path) -> Tuple[int, int]:
    tex_count = 0
    bib_count = 0

    # Case 1: extracted_root is a single file (.tex or .bib)
    if extracted_root.is_file():
        if extracted_root.suffix.lower() in ['.tex', '.bib']:
            dest_file = target_version_dir / extracted_root.name
            dest_file.parent.mkdir(parents = True, exist_ok = True)
            shutil.copy2(extracted_root, dest_file)
            if extracted_root.suffix.lower() == '.tex':
                tex_count += 1
                log(f"[SUCCESS] Copied single .tex file: {extracted_root.name}")
            else:
                bib_count += 1
                log(f"[SUCCESS] Copied single .bib file: {extracted_root.name}")
        return tex_count, bib_count

    # Case 2: extracted_root is a directory – walk recursively
    for root, dirs, files in os.walk(extracted_root):
        for fname in files:
            fpath = Path(root) / fname
            suffix = fpath.suffix.lower()
            if suffix in ('.tex', '.bib'):
                rel_path = fpath.relative_to(extracted_root)
                dest_path = target_version_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(fpath, dest_path)
                    if suffix == '.tex':
                        tex_count += 1
                        log(f"[SUCCESS] Copied .tex: {rel_path}")
                    else:
                        bib_count += 1
                        log(f"[SUCCESS] Copied .bib: {rel_path}")
                except Exception as e:
                    log(f"[ERROR] Failed to copy {fpath}: {e}")

    # Case 3: No .tex/.bib found in subdirectories → check root files
    if tex_count == 0 and bib_count == 0:
        for item in extracted_root.iterdir():
            if item.is_file() and item.suffix.lower() in ['.tex', '.bib']:
                try:
                    dest = target_version_dir / item.name
                    shutil.copy2(item, dest)
                    if item.suffix.lower() == '.tex':
                        tex_count += 1
                        log(f"[SUCCESS] Copied .tex directly: {item.name}")
                    else:
                        bib_count += 1
                        log(f"[SUCCESS] Copied .bib directly: {item.name}")
                except Exception as e:
                    log(f"[ERROR] Failed to copy {item}: {e}")
    log(f"[SUMMARY] Total copied: {tex_count} .tex files, {bib_count} .bib files")
    return tex_count, bib_count
