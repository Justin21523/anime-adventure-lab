#!/usr/bin/env python3
# scripts/batch_test.py
"""
Test script for batch processing functionality
"""
import os
import sys
import requests
import json
import time
from pathlib import Path

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_PREFIX = "/api/v1"


def submit_test_batch(job_type="caption", num_items=5):
    """Submit a test batch job"""
    api_url = f"{API_BASE_URL}{API_PREFIX}"

    # Create test inputs based on job type
    if job_type == "caption":
        inputs = [f"/tmp/test_image_{i}.jpg" for i in range(num_items)]
        config = {"max_length": 50, "num_beams": 3}
    elif job_type == "vqa":
        inputs = [
            {
                "image_path": f"/tmp/test_image_{i}.jpg",
                "question": f"What is in image {i}?",
            }
            for i in range(num_items)
        ]
        config = {"max_length": 100}
    elif job_type == "chat":
        inputs = [
            [{"role": "user", "content": f"Hello, this is test message number {i}"}]
            for i in range(num_items)
        ]
        config = {"max_length": 200, "temperature": 0.7}
    else:
        print(f"Unsupported job type: {job_type}")
        return None

    # Submit job
    payload = {"job_type": job_type, "inputs": inputs, "config": config}

    try:
        response = requests.post(f"{api_url}/batch/submit", json=payload, timeout=10)
        if response.status_code == 200:
            job_data = response.json()
            print(f"✅ Test batch job submitted successfully:")
            print(f"   Job ID: {job_data['job_id']}")
            print(f"   Task ID: {job_data['task_id']}")
            print(f"   Status: {job_data['status']}")
            print(f"   Total items: {job_data['total_items']}")
            return job_data
        else:
            print(f"❌ Failed to submit batch job: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error submitting batch job: {e}")
        return None


def monitor_job(job_id, max_wait=300):
    """Monitor job progress until completion"""
    api_url = f"{API_BASE_URL}{API_PREFIX}"
    start_time = time.time()

    print(f"\n🔍 Monitoring job {job_id}...")

    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{api_url}/batch/status/{job_id}", timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                status = status_data.get("status")
                processed = status_data.get("processed_items", 0)
                total = status_data.get("total_items", 0)
                failed = status_data.get("failed_items", 0)

                progress = (processed / total * 100) if total > 0 else 0

                print(
                    f"   Status: {status} | Progress: {progress:.1f}% ({processed}/{total}) | Failed: {failed}"
                )

                if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                    if status == "COMPLETED":
                        print(f"✅ Job completed successfully!")
                        if status_data.get("results_path"):
                            print(f"   Results: {status_data['results_path']}")
                    else:
                        print(f"❌ Job {status.lower()}")
                        if status_data.get("error_message"):
                            print(f"   Error: {status_data['error_message']}")
                    break

                time.sleep(5)
            else:
                print(f"❌ Failed to get job status: {response.status_code}")
                break
        except Exception as e:
            print(f"❌ Error monitoring job: {e}")
            break
    else:
        print(f"⏰ Job monitoring timed out after {max_wait} seconds")


def test_monitoring_endpoints():
    """Test monitoring API endpoints"""
    api_url = f"{API_BASE_URL}{API_PREFIX}"

    print("\n🔍 Testing monitoring endpoints...")

    endpoints = [
        ("/monitoring/health", "Health check"),
        ("/monitoring/metrics", "System metrics"),
        ("/monitoring/tasks", "Task metrics"),
        ("/monitoring/workers", "Worker status"),
    ]

    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{api_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {description}: OK")
            else:
                print(f"❌ {description}: {response.status_code}")
        except Exception as e:
            print(f"❌ {description}: Error - {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test batch processing system")
    parser.add_argument(
        "--job-type",
        choices=["caption", "vqa", "chat"],
        default="caption",
        help="Type of test job",
    )
    parser.add_argument("--num-items", type=int, default=5, help="Number of test items")
    parser.add_argument("--monitor", action="store_true", help="Monitor job progress")
    parser.add_argument(
        "--test-monitoring", action="store_true", help="Test monitoring endpoints"
    )

    args = parser.parse_args()

    if args.test_monitoring:
        test_monitoring_endpoints()

    # Submit test batch job
    job_data = submit_test_batch(args.job_type, args.num_items)

    if job_data and args.monitor:
        monitor_job(job_data["job_id"])


if __name__ == "__main__":
    main()
