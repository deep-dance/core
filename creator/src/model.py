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

import warnings
# warnings.filterwarnings('ignore')

import importlib
import time
import math
import numpy as np
import mdn
import json
import matplotlib.pyplot as plt
from data_utils import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from sklearn.model_selection import train_test_split

class DeepDanceModel:
    """deep.dance model class that represents a RNN with LSTM and MDN layers."""
    
    def __init__(
            self,
            look_back=10, lstm_layers=32, mdn_layers=3, validation_split=0.10,
            custom_loss=True, kinetic=False):
        self.validation_split = validation_split
        self.look_back = look_back
        self.mdn_layers = mdn_layers
        self.custom_loss = custom_loss
        self.trained = False
        dim = 52 if kinetic else 51
        
        self.model = keras.Sequential()
        self.model.add(layers.LSTM(
            lstm_layers,
            input_shape=(self.look_back, dim),  return_sequences=True))
        self.model.add(layers.LSTM(lstm_layers, return_sequences=True))
        self.model.add(layers.LSTM(lstm_layers))
        self.model.add(layers.Dense(lstm_layers))
        self.model.add(mdn.MDN(51, mdn_layers))

        loss_function = mdn.get_mixture_loss_func(51, self.mdn_layers)
        if self.custom_loss:
            loss_function = custom_mixture_loss_func(51, self.mdn_layers)
        
        self.model.compile(
            loss=loss_function,
            optimizer=keras.optimizers.Adam())
        self.model.summary()

    def train(self, x={}, y={}, epochs=10, batch_size=32):
        # Define callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10)
        callbacks = [keras.callbacks.TerminateOnNaN(), early_stopping]

        self.history = self.model.fit(
            x, y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=self.validation_split)

    def load(self, filename):
        loss_function = mdn.get_mixture_loss_func(51, self.mdn_layers)
        if self.custom_loss:
            loss_function = custom_mixture_loss_func(51, self.mdn_layers)

        self.model = tf.keras.models.load_model(
            filename,
            custom_objects = {
                'MDN': mdn.MDN,
                'mdn_loss_func': loss_function
            })

    def save(self, filename):
        self.model.save(filename)

    def generate(self, filename, x_test, seed, steps_limit, look_back=10,
        hip_correction=True, temperature=1.0, rescale_process=False, rescale_post=False,
        kinetic=False):
        
        kinetic_energy = np.array([x_test[seed][-1,-1] + 0.2 * x for x in range(steps_limit)])
        kinetic_energy_input = kinetic_energy if kinetic else []

        performance = generate_performance(
            self.model, x_test[seed], steps_limit=steps_limit, look_back=look_back,
            hip_correction=hip_correction, temp=temperature,
            rescale_process=rescale_process, rescale_post=rescale_post,
            kinetic_energy_input=kinetic_energy_input)
        save_seq_to_json(performance, filename)

    def evaluate(self, file_scores, file_loss):
        if hasattr(self, 'history'):
            # with open(file_scores, 'w') as fd:
            #     json.dump({'unknown': 0}, fd, indent=4)
            with open(file_scores, 'w') as fd:
                json.dump({
                    'end_loss': self.history.history['loss'][-1],
                    'end_val_loss': self.history.history['val_loss'][-1],
                }, fd, indent=4)

            with open(file_loss, 'w') as fd:
                zipper = zip(self.history.history['loss'],
                    self.history.history['val_loss'])
                json.dump({'perf': [{
                        'loss': l,
                        'val_loss': v,
                    } for l, v in zipper
                ]}, fd, indent=4)
        else:
            print('Could not access fitting history. ' + 
                'Place train model first and check custom loss function.')

            with open(file_scores, 'w') as fd:
                json.dump({'unknown': 0}, fd, indent=4)

        #save to file in json format so that it can be used by 
        # save_seq_to_json(longer_performance, "test_performance.json", path_base_dir=os.path.abspath("./"))
