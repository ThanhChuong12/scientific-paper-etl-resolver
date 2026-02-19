import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from workflow import run_scraper
from utils import log

if __name__ == "__main__":
    try:
        run_scraper()
    except KeyboardInterrupt:
        log("--- User interrupted. Shutting down. ---")
        sys.exit(0)
    except Exception as e:
        log(f"--- FATAL ERROR: {e} ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)