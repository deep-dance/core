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

epochs = 10

command = 'dvc exp run --queue \
-S train.dancers={0} \
-S train.validation_split={1} \
-S train.look_back={2} \
-S train.normalize_body={3} \
-S train.epochs={4} \
-S train.lstm_layer={5} \
-S generate.dancers={6} \
-S generate.validation_split={7} \
-S generate.look_back={8} \
-S generate.normalize_body={9} \
-S generate.temperature={10} \
-S generate.rescale_process={11} \
-S generate.rescale_post={12}'

epochs = 10
validation_split = 0.9
normalize_body = False

def run_exp_suite_01(call = False):
    dancers = 'all'
    for look_back in [70]:
        for lstm_layer in [32, 64, 128]:
            for temperature in [0.5, 0.75, 1.0]:
                for rescale_process in [True, False]:
                    for rescale_post in [True, False]:
                        if not(rescale_process and rescale_post):
                            sh_command = command.format(
                                # train
                                dancers,
                                validation_split,
                                look_back,
                                normalize_body,
                                epochs,
                                lstm_layer,
                                # generate
                                dancers,
                                validation_split,
                                look_back,
                                normalize_body,
                                temperature,
                                rescale_process,
                                rescale_post)
                            print(sh_command)
                            if call:
                                stream = os.popen(sh_command)
                                output = stream.read()
                                output
    print('dvc exp run --run-all --jobs 1')

run_exp_suite_01()




