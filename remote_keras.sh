#!/usr/bin/env bash
export BUCKET_NAME=agarflow-1773-ml
export JOB_NAME="agarflow_1773_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=europe-west1
export NUM_TRAIN_FILE=6

#for ((i=1;i<=$NUM_TRAIN_FILE;++i))
#do
#    gsutil -m -o GSUtil:parallel_composite_upload_threshold=150M cp train_data/train_data$i.npz  gs://$BUCKET_NAME/data/
#done

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir gs://$BUCKET_NAME/$JOB_NAME/ \
  --runtime-version 1.4 \
  --module-name trainer.network-keras \
  --package-path ./trainer \
  --region $REGION \
  --config=config.yaml \
  -- \
  --train-data-dir gs://$BUCKET_NAME/data/ \
  --num-train-file $NUM_TRAIN_FILE

#sudo shutdown -h now