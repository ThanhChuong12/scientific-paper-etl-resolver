import time
from threading import Lock
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import S2_DELAY

# Thread-safe request timing control
last_request_time = 0
request_lock = Lock()

# HTTP Session with Retry
session = requests.Session()

retry_strategy = Retry(
    total = 3,
    backoff_factor = 0.5,
    status_forcelist = [429, 500, 502, 503, 504],
)

adapter = HTTPAdapter(
    max_retries = retry_strategy,
    pool_connections = 10,
    pool_maxsize = 10
)

session.mount("http://", adapter)
session.mount("https://", adapter)

def enforce_rate_limit():
    global last_request_time
    with request_lock:
        current_time = time.time()
        elapsed = current_time - last_request_time
        wait_time = S2_DELAY - elapsed
        
        if wait_time > 0:
            time.sleep(wait_time)
        
        last_request_time = time.time()