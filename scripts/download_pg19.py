#!/usr/bin/env python3
"""Download PG-19 books as .txt files from Google Storage."""

import os
import sys
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://storage.googleapis.com/deepmind-gutenberg/"
RAW_DIR = "/home/x_x/Programming/LLM/pg19_raw/data"
OUT_DIR = "/home/x_x/Programming/LLM/pg19"
SPLITS = ["train", "validation", "test"]
MAX_WORKERS = 32

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
session.mount("https://", adapter)

def download_file(rel_path: str) -> tuple[str, bool, str]:
    url = BASE_URL + rel_path
    out_path = os.path.join(OUT_DIR, rel_path)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return rel_path, True, "skip"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for attempt in range(3):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(r.text)
            return rel_path, True, "ok"
        except Exception as e:
            if attempt == 2:
                return rel_path, False, str(e)
            time.sleep(2 ** attempt)

def main():
    all_files = []
    for split in SPLITS:
        list_path = os.path.join(RAW_DIR, f"{split}_files.txt")
        with open(list_path) as f:
            files = [line.strip() for line in f if line.strip()]
        all_files.extend(files)
        print(f"{split}: {len(files)} books")

    total = len(all_files)
    print(f"\nTotal: {total} books -> {OUT_DIR}\n")

    done = 0
    failed = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_file, p): p for p in all_files}
        for future in as_completed(futures):
            rel_path, ok, status = future.result()
            with lock:
                done += 1
                if not ok:
                    failed.append((rel_path, status))
                if done % 500 == 0 or done == total:
                    pct = done / total * 100
                    print(f"  {done}/{total} ({pct:.1f}%)  failures: {len(failed)}")
                    sys.stdout.flush()

    print(f"\nDone. {total - len(failed)}/{total} succeeded.")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for path, err in failed[:20]:
            print(f"  {path}: {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

if __name__ == "__main__":
    main()
