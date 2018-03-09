#!/usr/bin/env bash
export BUCKET_NAME=agarflow-1773-ml
export JOB_NAME="agarflow_1773_$(date +%Y%m%d_%H%M%S)"

gcloud ml-engine local train \
  --module-name trainer.network-keras \
  --package-path ./trainer \
  -- \
  --train-file combined_raw.npy \
  --job-dir ./tmp/ \
  --augmented False