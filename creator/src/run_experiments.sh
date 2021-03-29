#!/bin/bash

# --------------------------
# Smoke test(s)
# --------------------------
# dvc exp run --queue \
#     -S train.epochs=1 \
#     -S train.custom_loss=False \
#     -S generate.custom_loss=False

# --------------------------
# Queue up experiments
# --------------------------
# look_back = [10, 30, 100, 300]
# lstm_layer = [32, 64, 128]
# mdn_layer = [2, 3, 5]
# --------------------------
dvc exp run --queue \
    -S train.epochs=10 \
    -S train.look_back=10 \
    -S train.lstm_layer=32

dvc exp run --queue \
    -S train.epochs=10 \
    -S train.look_back=10 \
    -S train.lstm_layer=64

dvc exp run --queue \
    -S train.epochs=10 \
    -S train.look_back=10 \
    -S train.lstm_layer=128
# ---------------------------
dvc exp run --queue \
    -S train.epochs=10 \
    -S train.look_back=30 \
    -S train.lstm_layer=32

dvc exp run --queue \
    -S train.epochs=10 \
    -S train.look_back=30 \
    -S train.lstm_layer=64

dvc exp run --queue \
    -S train.epochs=10 \
    -S train.look_back=30 \
    -S train.lstm_layer=128
# ---------------------------
dvc exp run --queue \
    -S train.epochs=10 \
    -S train.look_back=100 \
    -S train.lstm_layer=32

dvc exp run --queue \
    -S train.epochs=10 \
    -S train.look_back=100 \
    -S train.lstm_layer=64

dvc exp run --queue \
    -S train.epochs=10 \
    -S train.look_back=100 \
    -S train.lstm_layer=128
# ---------------------------
dvc exp run --queue \
    -S train.epochs=10 \
    -S train.look_back=300 \
    -S train.lstm_layer=32

dvc exp run --queue \
    -S train.epochs=10 \
    -S train.look_back=300 \
    -S train.lstm_layer=64

dvc exp run --queue \
    -S train.epochs=10 \
    -S train.look_back=300 \
    -S train.lstm_layer=128
# ---------------------------

# ---------------------------
# Run queued experiments
# ---------------------------
dvc exp run --run-all
# Or run in parallel
# dvc exp run --run-all --jobs 2