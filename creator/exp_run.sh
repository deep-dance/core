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
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=70 -S train.lstm_layer=32 -S train.mdn_layer=2 -S train.dancers=all -S generate.look_back=70 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=70 -S train.lstm_layer=32 -S train.mdn_layer=3 -S train.dancers=all -S generate.look_back=70 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=70 -S train.lstm_layer=32 -S train.mdn_layer=5 -S train.dancers=all -S generate.look_back=70 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=70 -S train.lstm_layer=64 -S train.mdn_layer=2 -S train.dancers=all -S generate.look_back=70 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=70 -S train.lstm_layer=64 -S train.mdn_layer=3 -S train.dancers=all -S generate.look_back=70 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=70 -S train.lstm_layer=64 -S train.mdn_layer=5 -S train.dancers=all -S generate.look_back=70 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=70 -S train.lstm_layer=128 -S train.mdn_layer=2 -S train.dancers=all -S generate.look_back=70 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=70 -S train.lstm_layer=128 -S train.mdn_layer=3 -S train.dancers=all -S generate.look_back=70 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=70 -S train.lstm_layer=128 -S train.mdn_layer=5 -S train.dancers=all -S generate.look_back=70 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=32 -S train.mdn_layer=2 -S train.dancers=all -S generate.look_back=10 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=32 -S train.mdn_layer=3 -S train.dancers=all -S generate.look_back=10 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=32 -S train.mdn_layer=5 -S train.dancers=all -S generate.look_back=10 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=64 -S train.mdn_layer=2 -S train.dancers=all -S generate.look_back=10 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=64 -S train.mdn_layer=3 -S train.dancers=all -S generate.look_back=10 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=64 -S train.mdn_layer=5 -S train.dancers=all -S generate.look_back=10 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=128 -S train.mdn_layer=2 -S train.dancers=all -S generate.look_back=10 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=128 -S train.mdn_layer=3 -S train.dancers=all -S generate.look_back=10 -S generate.dancers=all
dvc exp run --queue -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=128 -S train.mdn_layer=5 -S train.dancers=all -S generate.look_back=10 -S generate.dancers=all
dvc exp run --run-all --jobs 1


# Run queued experiments
# ---------------------------
# Or run in parallel
# dvc exp run --run-all --jobs 2