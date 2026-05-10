#!/bin/bash
# dev.sh - Native local development startup. This script intentionally avoids Docker.

set -e

if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
fi

export START_WORKER="${START_WORKER:-0}"
export START_REDIS="${START_REDIS:-0}"

exec ./run_local.sh
