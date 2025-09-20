#!/usr/bin/env python3
# scripts/monitoring_setup.py
"""
Initialize monitoring and metrics collection
"""
import os
import sys
import time
import threading
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.shared_cache import get_shared_cache
from core.monitoring.metrics import MetricsCollector
from core.monitoring.logger import structured_logger


def setup_monitoring():
    """Setup monitoring system"""
    get_shared_cache()

    # Initialize metrics collector
    metrics_collector = MetricsCollector()
    structured_logger.info("Metrics collector initialized")

    return metrics_collector


def start_metrics_collection(metrics_collector, interval=30):
    """Start continuous metrics collection"""

    def collect_loop():
        while True:
            try:
                metrics_collector.record_system_metrics()
                structured_logger.debug("System metrics recorded")
                time.sleep(interval)
            except KeyboardInterrupt:
                structured_logger.info("Metrics collection stopped")
                break
            except Exception as e:
                structured_logger.error(f"Error collecting metrics: {e}")
                time.sleep(interval)

    # Start collection in background thread
    collection_thread = threading.Thread(target=collect_loop, daemon=True)
    collection_thread.start()
    structured_logger.info(f"Started metrics collection (interval: {interval}s)")

    return collection_thread


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Setup monitoring system")
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=30,
        help="Metrics collection interval in seconds",
    )
    parser.add_argument(
        "--daemon", "-d", action="store_true", help="Run as daemon process"
    )

    args = parser.parse_args()

    # Setup monitoring
    metrics_collector = setup_monitoring()

    if args.daemon:
        # Start as daemon
        collection_thread = start_metrics_collection(metrics_collector, args.interval)
        try:
            # Keep main thread alive
            while collection_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            structured_logger.info("Monitoring daemon stopped")
    else:
        # Run once and exit
        metrics_collector.record_system_metrics()
        structured_logger.info("Metrics recorded successfully")


if __name__ == "__main__":
    main()
