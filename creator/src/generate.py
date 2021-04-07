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

import os
import argparse
import yaml

# Filter out INFO & WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.model_selection import train_test_split

from data_utils import *
from model import DeepDanceModel

def get_parser():
	parser = argparse.ArgumentParser(description="Generate new deep.dance sequences.")
	return parser

if __name__ == '__main__':
    params = yaml.safe_load(open('params.yaml'))['generate']
    dancers = params['dancers']
    tags = params['tags']
    seed = params['seed']
    steps_limit = params['steps_limit']
    look_back = params['look_back']
    random_state = params['random_state']
    custom_loss = params['custom_loss']
    validation_split = params['validation_split']
    test_size = params['test_size']

    args = get_parser().parse_args()

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # TODO Uncomment and fix NPZ export in prepare stage.
    #
    # print("Reading prepared data from file. This might take a while..")
    # x_test = np.load('../data/train/prepared/x_test.npz')

    # TODO Remove after NPZ export fix
    print("Loading prepared data. This might take a while..")
    # x, y = get_training_data(look_back = look_back)
    selected_dancers = dancers.split(',')
    for d in selected_dancers:
        d = d.strip()

    selected_tags = tags.split(',')
    for t in selected_tags:
        t = d.strip()

    x, y = get_training_data(dancers = selected_dancers, tags = selected_tags,
        look_back = look_back)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, shuffle=True, random_state=random_state)

    os.makedirs(os.path.join('../', 'data', 'generated'), exist_ok=True)

    model = DeepDanceModel(custom_loss=custom_loss)
    model.load('../data/models/deep-dance.h5')

    print('Generating sequences...')
    model.generate('../data/generated/deep-dance-seq.json', x_test,
        seed, steps_limit)

    
    
    
    
