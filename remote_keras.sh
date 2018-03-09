#!/usr/bin/env bash
export BUCKET_NAME=agarflow-1773-ml
export JOB_NAME="agarflow_1773_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=europe-west1

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir gs://$BUCKET_NAME/$JOB_NAME/ \
  --runtime-version 1.4 \
  --module-name trainer.network-keras \
  --package-path ./trainer \
  --region $REGION \
  --config=config.yaml \
  -- \
  --train-file gs://agarflow-1773-ml/combined_raw.npy \
  --augmented True