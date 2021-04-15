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
    
    random_state = params['random_state']
    test_size = params['test_size']
    validation_split = params['validation_split']
    look_back = params['look_back']
    normalize_body = params['normalize_body']
    hip_correction = params['hip_correction']
    kinetic = params['kinetic']
    
    seed = params['seed']
    steps_limit = params['steps_limit']

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
    print('seed:', seed)
    print('steps_limit:', steps_limit)
    print('-------------------------')

    args = get_parser().parse_args()

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # TODO Uncomment and fix NPZ export in prepare stage.
    #
    # print("Reading prepared data from file. This might take a while..")
    # x_test = np.load('../data/train/prepared/x_test.npz')

    # TODO Remove after NPZ export fix
    print("Loading prepared data. This might take a while...")

    selected_dancers = stringlist_to_array(dancers)
    selected_tags = stringlist_to_array(tags)

    x, y = get_training_data(dancers=selected_dancers, tags=selected_tags,
        look_back=look_back, normalize_body=normalize_body,
        hip_correction=hip_correction, add_kinetic_energy=kinetic)
        
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, shuffle=True, random_state=random_state)

    os.makedirs(os.path.join('../', 'data', 'generated'), exist_ok=True)

    model = DeepDanceModel(kinetic=kinetic)
    model.load('../data/models/deep-dance.h5')

    print('Generating sequences...')
    for temperature in [0.5, 0.75, 1.0]:
        for rescale_process in [False]:
            for rescale_post in [True]:
                if not(rescale_process and rescale_post):
                    postfix = str(temperature).replace('.', '_') + '-' + str(rescale_process) + '-' + str(rescale_post)
                    model.generate('../data/generated/deep-dance-seq-' + postfix + '.json',
                        x_test, seed, steps_limit, look_back, hip_correction,
                        temperature, rescale_process, rescale_post)

    
    
    
    
