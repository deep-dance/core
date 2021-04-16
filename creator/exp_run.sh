#!/bin/bash


# --------------------------
# Smoke test(s)
# --------------------------
# dvc exp run --queue \
#     -S train.epochs=1 \
#     -S train.custom_loss=False \
#     -S generate.custom_loss=False

# --------------------------
# Queue up experiments. Please see 'exp_generate.py' on how to generate these DVC calls.
# --------------------------
dvc exp run --queue -S train.dancers=all -S train.epochs=10 -S generate.dancers=all


# Run queued experiments
# ---------------------------
# Or run in parallel
# dvc exp run --run-all --jobs 2