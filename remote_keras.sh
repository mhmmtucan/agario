#!/usr/bin/env bash
export BUCKET_NAME=agarflow-1773-ml
export JOB_NAME="agarflow_1773_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=europe-west1

#gsutil -m -o GSUtil:parallel_composite_upload_threshold=150M cp train_data.npy  gs://$BUCKET_NAME/data/

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir gs://$BUCKET_NAME/$JOB_NAME/ \
  --runtime-version 1.4 \
  --module-name trainer.network-keras \
  --package-path ./trainer \
  --region $REGION \
  --config=config.yaml \
  -- \
  --train-file gs://$BUCKET_NAME/data/train_data.npy \
  --augmented False