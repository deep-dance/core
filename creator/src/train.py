# This file is part of deep-dance.

# deep-dance is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# deep-dance is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with deep-dance.  If not, see <https://www.gnu.org/licenses/>.

# ---------------------------------------------------------------------
# Params for training:
#
# data = ['all']
# epochs = [10]
# batch_size = [128]
# look_back = [10, 30, 100, 300]
# lstm_layer = [32, 64, 128]
# mdn_layer = [2, 3, 5]
# ->
# folder with models, params, sequences; naming based on params
# value_losses
# final_value_loss
# ---------------------------------------------------------------------

import os
import argparse
import yaml

# Filter out INFO & WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Data import workaround
import numpy as np
from data_utils import *
from sklearn.model_selection import train_test_split

from model import DeepDanceModel

def get_parser():
	parser = argparse.ArgumentParser(description="Train a deep.dance model.")
	return parser

if __name__ == '__main__':
    params = yaml.safe_load(open('params.yaml'))['train']
    dancers = params['dancers']
    tags = params['tags']
    random_state = params['random_state']
    test_size = params['test_size']
    validation_split = params['validation_split']
    epochs = params['epochs']
    batch_size = params['batch_size']
    look_back = params['look_back']
    lstm_layer = params['lstm_layer']
    mdn_layer = params['mdn_layer']
    normalize_body = params['normalize_body']
    hip_correction = params['hip_correction']
    kinetic = params['kinetic']

    print('-------------------------')
    print('dancers:', dancers)
    print('tags:', tags)
    print('random_state:', random_state)
    print('test_size:', test_size)
    print('validation_split:', validation_split)
    print('look_back:', look_back)
    print('normalize_body:', normalize_body)
    print('hip_correction:', hip_correction)
    print('kinetic:', kinetic)
    print('epochs:', epochs)
    print('batch_size:', batch_size)
    print('lstm_layer:', lstm_layer)
    print('mdn_layer:', mdn_layer)
    print('-------------------------')

    args = get_parser().parse_args()

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))

    # TODO Uncomment and fix NPZ export in prepare stage.
    #
    # print("Reading prepared data from file. This might take a while..")
    # x = np.load('../data/train/prepared/x.npz')
    # x_train = np.load('../data/train/prepared/x_train.npz')
    # x_test = np.load('../data/train/prepared/x_test.npz')

    # y = np.load('../data/train/prepared/y.npz')
    # y_train = np.load('../data/train/prepared/y_train.npz')
    # y_test = np.load('../data/train/prepared/y_test.npz')

    # TODO Remove after NPZ export fix
    print("Loading prepared data. This might take a while...")

    selected_dancers = stringlist_to_array(dancers)
    selected_tags = stringlist_to_array(tags)

    x, y = get_training_data(dancers=selected_dancers, tags=selected_tags,
        look_back=look_back, normalize_body=normalize_body,
        hip_correction=hip_correction, add_kinetic_energy=kinetic)
    
    print("Data loaded. Splitting now...")
    x_train, x_test, y_train, y_test=train_test_split(
            x, y, test_size=test_size, shuffle=True, random_state=random_state)
    

    print("Data shape(s):")
    print("x: ", np.shape(x_train))
    print("x_train: ", np.shape(x_train))
    print("x_test: ", np.shape(x_test))
    print("y: ", np.shape(x_train))
    print("y_train: ", np.shape(y_train))
    print("y_test: ", np.shape(y_test))


    os.makedirs(os.path.join('../', 'data', 'models'), exist_ok=True)
    os.makedirs(os.path.join('../', 'data', 'metrics'), exist_ok=True)
 

    model = DeepDanceModel(
        look_back=look_back, lstm_layers=lstm_layer,
        mdn_layers=mdn_layer, validation_split=validation_split,
        kinetic=kinetic)
    model.train(x, y, epochs = epochs, batch_size=batch_size)

    print('Saving model and metrics...')
    model.save('../data/models/deep-dance.h5')
    model.evaluate('../data/metrics/deep-dance-scores.json',
        '../data/metrics/deep-dance-loss.json')
    
    
    
    
    
