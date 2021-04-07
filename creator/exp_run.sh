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
echo "Training started..."

dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=32 -S train.mdn_layer=2 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=10
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=32 -S train.mdn_layer=3 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=10
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=32 -S train.mdn_layer=5 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=10
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=64 -S train.mdn_layer=2 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=10
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=64 -S train.mdn_layer=3 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=10
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=64 -S train.mdn_layer=5 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=10
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=128 -S train.mdn_layer=2 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=10
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=128 -S train.mdn_layer=3 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=10
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=10 -S train.lstm_layer=128 -S train.mdn_layer=5 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=10
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=100 -S train.lstm_layer=32 -S train.mdn_layer=2 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=100
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=100 -S train.lstm_layer=32 -S train.mdn_layer=3 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=100
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=100 -S train.lstm_layer=32 -S train.mdn_layer=5 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=100
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=100 -S train.lstm_layer=64 -S train.mdn_layer=2 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=100
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=100 -S train.lstm_layer=64 -S train.mdn_layer=3 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=100
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=100 -S train.lstm_layer=64 -S train.mdn_layer=5 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=100
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=100 -S train.lstm_layer=128 -S train.mdn_layer=2 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=100
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=100 -S train.lstm_layer=128 -S train.mdn_layer=3 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=100
dvc exp run -S train.epochs=10 -S train.batch_size=32 -S train.look_back=100 -S train.lstm_layer=128 -S train.mdn_layer=5 -S train.dancers=girish,maria,mark,marlen,raymond,tinyeung -S train.tags=impro -S generate.look_back=100

echo "Training finished."
echo "$now"
# ---------------------------
# Run queued experiments
# ---------------------------
# dvc exp run --run-all
# Or run in parallel
# dvc exp run --run-all --jobs 2