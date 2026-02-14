import os
import psutil
import time
import threading
from datetime import datetime
from pathlib import Path

# Helpers 
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def arxiv_id_to_folder(id_str: str) -> str:
    return id_str.replace('.', '-')

def format_yymm_id(arxiv_id: str) -> str:
    base = arxiv_id.split('v')[0]
    return base.replace('.', '-')

def get_total_size(path_root: Path) -> int:
    total = 0
    try:
        if not path_root.exists():
            return 0
        for p in path_root.rglob('*'):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except Exception:
                pass
    except Exception:
        pass
    return total

def track_memory_usage(stop_event, mem_stats):
    process = psutil.Process(os.getpid())
    samples = []
    while not stop_event.is_set():
        try:
            rss = process.memory_info().rss / (1024 * 1024)  # MB
            samples.append(rss)
        except:
            pass
        time.sleep(0.1)

    if samples:
        mem_stats["max"] = max(samples)
        mem_stats["avg"] = sum(samples) / len(samples)
    else:
        mem_stats["max"] = 0
        mem_stats["avg"] = 0