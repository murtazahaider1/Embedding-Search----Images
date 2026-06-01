"""
refresh_catalog.py

Runs build_index.py --refresh on a schedule so your catalog
stays in sync with the live Zarr feed automatically.

Two modes:
  python refresh_catalog.py --every 24h     # runs every 24 hours (default)


"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def run_refresh():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] Starting catalog refresh …")
    result = subprocess.run(
        [sys.executable, "build_index.py", "--refresh"],
        capture_output=False,   # let output stream to terminal/log
    )
    if result.returncode == 0:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Refresh complete ✓")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Refresh FAILED (exit {result.returncode})")


def parse_interval(s: str) -> int:
    """Parse '6h', '30m', '1d' into seconds."""
    s = s.strip().lower()
    if s.endswith("d"):
        return int(s[:-1]) * 86400
    if s.endswith("h"):
        return int(s[:-1]) * 3600
    if s.endswith("m"):
        return int(s[:-1]) * 60
    return int(s)   # assume seconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Periodic Zarr catalog refresh")
    parser.add_argument(
        "--every",
        default="24h",
        metavar="INTERVAL",
        help="Refresh interval: e.g. 6h, 12h, 24h, 30m (default: 24h)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single refresh immediately and exit",
    )
    args = parser.parse_args()

    if args.once:
        run_refresh()
        sys.exit(0)

    interval = parse_interval(args.every)
    print(f"Catalog auto-refresh every {args.every} ({interval}s)")
    print("Press Ctrl+C to stop.\n")

    while True:
        run_refresh()
        next_run = datetime.fromtimestamp(time.time() + interval)
        print(f"Next refresh at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(interval)


