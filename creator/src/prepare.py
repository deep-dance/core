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
from os import path
import argparse
import subprocess
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

from data_utils import *

if __name__ == '__main__':
    params = yaml.safe_load(open('params.yaml'))['prepare']

    # Test data set split ratio
    split = params['split']
    look_back = params['look_back']
    random_state = params['random_state']

    print("split param:", split)
    print("look_back param:", look_back)

    output = {}
    output['test'] = True

    os.makedirs(os.path.join('../', 'data', 'train', 'prepared'), exist_ok=True)

    print("Running dataset preparation and compression. This might take while...")

    X, y = get_training_data(look_back = look_back)

    # split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, shuffle=True, random_state=random_state)
    
    np.savez_compressed('../data/train/prepared/x', data=X)
    np.savez_compressed('../data/train/prepared/y', data=y)
    np.savez_compressed('../data/train/prepared/x_train', data=X_train)
    np.savez_compressed('../data/train/prepared/x_test', data=X_test)
    np.savez_compressed('../data/train/prepared/y_train', data=y_train)
    np.savez_compressed('../data/train/prepared/y_test', data=y_test)

    
