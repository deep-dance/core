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
dvc exp run --queue -S train.dancers=all -S train.validation_split=0.9 -S train.look_back=70 -S train.epochs=10 -S train.lstm_layer=32 -S generate.dancers=all -S generate.validation_split=0.9 -S generate.look_back=70


# Run queued experiments
# ---------------------------
# Or run in parallel
# dvc exp run --run-all --jobs 2