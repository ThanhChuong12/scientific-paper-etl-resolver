from pathlib import Path

# ---------------- CONFIG ----------------
SEMANTIC_SCHOLAR_API_KEY = "YNsSffUP0ca1D7Rh0CBzS5XMot1LxNji24FP9dHi"
STUDENT_ID = "23120195"

BASE_DIR = Path(STUDENT_ID) # Base output path
START_MONTH = "2412"        # Format YYMM
START_ID = 15272            # arXiv ID range start
# END_ID = 20270              # arXiv ID range end
END_ID = 15274

S2_DELAY = 0.5              # Minimum delay between API calls
MAX_WORKERS = 5             # Number of parallel worker threads
BATCH_SIZE = 50             # Number of papers per batch

# Supported image file extensions
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.pdf', '.eps', '.svg', '.bmp', '.tiff', '.gif', '.ico'}

# HTTP Headers
ARXIV_EPRINT_HEADERS = {
    "User-Agent": "lab-scraper/1.0 (your_email@example.com)"
}

SEMANTIC_SCHOLAR_HEADERS = {
    "x-api-key": SEMANTIC_SCHOLAR_API_KEY,
    "User-Agent": "Academic-Research-Crawler/1.0 (for educational use)"
}