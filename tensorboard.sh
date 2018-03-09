#!/usr/bin/env bash
tensorboard --logdir=${JOB_DIR}/logs --port 8000 --reload_interval=5