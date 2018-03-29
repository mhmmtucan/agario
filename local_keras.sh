#!/usr/bin/env bash
export BUCKET_NAME=agarflow-1773-ml
export JOB_NAME="agarflow_1773_$(date +%Y%m%d_%H%M%S)"
export NUM_TRAIN_FILE=4

gcloud ml-engine local train \
  --module-name trainer.network-keras \
  --package-path ./trainer \
  -- \
  --train-data-dir train_data/ \
  --job-dir ./tmp/ \
  --num-train-file $NUM_TRAIN_FILE