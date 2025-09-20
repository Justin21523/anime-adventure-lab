#!/usr/bin/env python3
# scripts/start_worker.py
"""
Celery worker startup script with proper environment setup
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Shared cache bootstrap
from core.shared_cache import get_shared_cache

get_shared_cache()


def start_worker(concurrency: int = None, queues: str = None, loglevel: str = "INFO"):
    """Start Celery worker with specified configuration"""
    from workers.celery_app import celery_app

    # Build worker command
    cmd_args = [
        "worker",
        f"--loglevel={loglevel}",
        "--without-gossip",
        "--without-mingle",
        "--without-heartbeat",
    ]

    if concurrency:
        cmd_args.append(f"--concurrency={concurrency}")

    if queues:
        cmd_args.append(f"--queues={queues}")
    else:
        # Default to all queues
        cmd_args.append("--queues=default,vision,text,training")

    print(f"Starting Celery worker with args: {' '.join(cmd_args)}")

    # Start worker
    celery_app.worker_main(cmd_args)


def main():
    parser = argparse.ArgumentParser(description="Start Celery worker")
    parser.add_argument(
        "--concurrency", "-c", type=int, help="Number of concurrent worker processes"
    )
    parser.add_argument(
        "--queues", "-Q", type=str, help="Comma-separated list of queues to consume"
    )
    parser.add_argument(
        "--loglevel",
        "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    start_worker(
        concurrency=args.concurrency, queues=args.queues, loglevel=args.loglevel
    )


if __name__ == "__main__":
    main()
