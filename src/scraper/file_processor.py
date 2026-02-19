"""
Downloader Module
Handles downloading and extracting LaTeX source files from arXiv
"""
import requests
import tarfile
import gzip
import struct
import os
import time
from arxiv_client import format_arxiv_id, get_arxiv_metadata
from metadata_collector import build_metadata_and_refs
import threading
arxiv_download_lock = threading.Lock()

def download_and_extract_tex_bib(arxiv_id, version, base_save_dir="./downloads", keep_exts=("tex", "bib")):
    # ...existing code...
    """
    Download and extract .tex and .bib files from arXiv source
    
    Args:
        arxiv_id: ArXiv ID without version
        version: Version number
        base_save_dir: Base directory to save files
        keep_exts: Tuple of extensions to keep (default: tex, bib)
        
    Returns:
        tuple: (success, size_before, size_after, is_pdf)
            - success: True if extraction successful
            - size_before: Total size of all files in archive (bytes)
            - size_after: Total size of kept files (bytes)
            - is_pdf: True if source is PDF (no LaTeX available)
    """
    safe_id = format_arxiv_id(arxiv_id)
    save_dir = os.path.join(base_save_dir, safe_id, "tex", f"{safe_id}v{version}")
    os.makedirs(save_dir, exist_ok=True)

    source_url = f"https://arxiv.org/e-print/{arxiv_id}v{version}"
    temp_path = os.path.join(save_dir, f"{safe_id}v{version}.tmp")

    try:
        r = requests.get(source_url, timeout=60)
        r.raise_for_status()
        
        with open(temp_path, "wb") as f:
            f.write(r.content)

        size_before = 0
        size_after = 0

        # Try tar.gz first
        try:
            with tarfile.open(temp_path, "r:gz") as tar:
                # Calculate size_before (all files)
                for member in tar.getmembers():
                    if member.isfile():
                        size_before += member.size

                # Extract only keep_exts files
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    
                    filename = os.path.basename(member.name)
                    ext = filename.split(".")[-1].lower()
                    
                    if ext in keep_exts:
                        target_path = os.path.join(save_dir, member.name)
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        
                        with tar.extractfile(member) as src, open(target_path, "wb") as dst:
                            dst.write(src.read())
                        
                        size_after += member.size

        except tarfile.ReadError:
            # Try single gzip file
            try:
                with gzip.open(temp_path, 'rb') as gz:
                    # Read and skip header to find filename
                    gz.seek(0)
                    magic = gz.read(2)
                    if magic != b'\x1f\x8b':
                        raise ValueError("Not a gzip file")
                    
                    gz.read(1)  # compression method
                    flags = struct.unpack('B', gz.read(1))[0]
                    gz.read(6)  # mtime, xfl, os
                    
                    filename = None
                    if flags & 0x08:  # FNAME flag
                        fname_bytes = b''
                        while True:
                            byte = gz.read(1)
                            if byte == b'\x00' or not byte:
                                break
                            fname_bytes += byte
                        filename = fname_bytes.decode('latin-1', errors='ignore')
                    
                    # Read content
                    gz.seek(0)
                    content = gz.read()
                    size_before = len(content)
                    
                    # Generate filename if not found
                    if not filename:
                        filename = f"{arxiv_id}v{version}.tex"
                    
                    # Normalize filename
                    filename_base = os.path.basename(filename)
                    if '.' in filename_base:
                        parts = filename_base.split('.')
                        if len(parts[0].split('-')) == 1 and parts[0].count('.') > 0:
                            filename_base = filename_base.replace('.', '-', 1)
                    
                    # Check extension
                    ext = filename_base.split(".")[-1].lower()
                    if ext in keep_exts:
                        save_path = os.path.join(save_dir, filename_base)
                        with open(save_path, 'wb') as f:
                            f.write(content)
                        size_after = len(content)
                    else:
                        print(f"⚠️  Skipping file with extension: .{ext}")
                        os.remove(temp_path)
                        return False, 0, 0, False

            except Exception as e2:
                content_type = r.headers.get('Content-Type', '')
                if 'pdf' in content_type.lower():
                    print(f"⚠️  Source is PDF (no LaTeX available)")
                    os.remove(temp_path)
                    return False, 0, 0, True  # is_pdf = True
                else:
                    print(f"⚠️  Unknown format: {e2}")
                os.remove(temp_path)
                return False, 0, 0, True

        os.remove(temp_path)
        return True, size_before, size_after, False

    except Exception as e:
        print(f"❌ Download failed: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, 0, 0, False


def download_paper(arxiv_id, config, config_path="config.json", collect_metrics=False):
    """
    Download all versions and metadata for a single paper
    
    Args:
        arxiv_id: ArXiv ID without version
        config: Configuration dictionary
        config_path: Path to config file (for progress tracking)
        collect_metrics: Whether to collect and return metrics
        
    Returns:
        tuple: (success, metrics_dict) if collect_metrics=True, else bool
    """
    settings = config["download_settings"]
    safe_id = format_arxiv_id(arxiv_id)
    paper_dir = os.path.join(settings["base_dir"], safe_id)
    
    # Initialize metrics
    metrics = {
        'size_before': 0,
        'size_after': 0,
        'num_references': 0,
        'num_versions': 0,
        'reference_fetch_success': False,
        'no_tex_source': False
    }
    
    try:
        # Get ALL metadata from arXiv (ONLY ONCE!)
        print(f"🔍 Fetching metadata from arXiv webpage...")
        with arxiv_download_lock:
            arxiv_metadata = get_arxiv_metadata(arxiv_id)
            time.sleep(0.5)
        latest_v = arxiv_metadata.get('latest_version', 0)
        
        if latest_v == 0:
            print(f"❌ Paper {arxiv_id} not found.")
            raise ValueError(f"Paper {arxiv_id} not found.")
            
        print(f"\n🔍 {arxiv_id}: found {latest_v} versions")
        metrics['num_versions'] = latest_v
        
        # Download all versions and accumulate sizes
        total_size_before = 0
        total_size_after = 0
        has_pdf_version = False  # Track if any version is PDF-only
        
        for v in range(1, latest_v + 1):
            success, size_before, size_after, is_pdf = download_and_extract_tex_bib(
                arxiv_id, 
                v, 
                base_save_dir=settings["base_dir"],
                keep_exts=tuple(settings["keep_extensions"])
            )
            if success:
                total_size_before += size_before
                total_size_after += size_after
                print(f"   Version {v}: {size_before} bytes before → {size_after} bytes after")
            else:
                if is_pdf:
                    has_pdf_version = True
                print(f"⚠️  Skipping version {v}")
            time.sleep(settings["delay_between_versions"])
        
        # Mark as no_tex_source if any version is PDF-only
        metrics['no_tex_source'] = has_pdf_version
        metrics['size_before'] = total_size_before
        metrics['size_after'] = total_size_after
        print(f"📊 Total for {arxiv_id}: {total_size_before} bytes before → {total_size_after} bytes after")
        
        # Download metadata and references (pass arxiv_metadata)
        print(f"📥 Downloading metadata and references for {arxiv_id}...")
        api_key = config.get("api_keys", {}).get("semantic_scholar")
        metadata = build_metadata_and_refs(
            arxiv_id, 
            arxiv_metadata=arxiv_metadata,
            output_dir=settings["base_dir"],
            api_key=api_key
        )
        
        # Count references and check if fetch was successful
        if metadata and 'references' in metadata:
            metrics['num_references'] = len(metadata['references'])
        
        # Check if reference fetch was successful (all_references_count > 0)
        if metadata and metadata.get('all_references_count', 0) > 0:
            metrics['reference_fetch_success'] = True
        
        # Mark as completed
        if safe_id not in config["progress"]["completed_papers"]:
            config["progress"]["completed_papers"].append(safe_id)
        
        print(f"✅ Successfully downloaded {arxiv_id}")
        
        if collect_metrics:
            return True, metrics
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {arxiv_id}: {e}")
        
        # Mark as failed
        if safe_id not in config["progress"]["failed_papers"]:
            config["progress"]["failed_papers"].append(safe_id)
        
        if collect_metrics:
            return False, metrics
        return False

def _get_directory_size(path):
    """
    Calculate total size of directory in bytes
    
    Args:
        path: Directory path
        
    Returns:
        Total size in bytes
    """
    total = 0
    try:
        if not os.path.exists(path):
            return 0
            
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    if os.path.exists(filepath):
                        total += os.path.getsize(filepath)
                except (OSError, PermissionError):
                    pass
    except Exception:
        pass
    return total