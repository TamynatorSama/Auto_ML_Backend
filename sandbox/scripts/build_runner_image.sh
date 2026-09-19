#!/usr/bin/env bash
# Builds automl-runner:<runtime hash>, tags it automl-runner:current, prints the hash.
set -euo pipefail
cd "$(dirname "$0")/../.."

HASH=$(python3 sandbox/scripts/runtime_hash.py)
docker build -f code_gen_eval/Dockerfile.runner --build-arg RUNTIME_HASH="$HASH" \
  -t "automl-runner:$HASH" -t automl-runner:current .
echo "$HASH"
