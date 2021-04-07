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
-S train.epochs={0} \
-S train.batch_size={1} \
-S train.look_back={2} \
-S train.lstm_layer={3} \
-S train.mdn_layer={4} \
-S train.dancers={5} \
-S train.tags={6} \
-S generate.look_back={7}'

def run_exp_suite_01(call = False):
    dancers = 'girish,maria,mark,marlen,raymond,tinyeung'
    tags = 'impro'
    for batch_size in [32]:
        for look_back in [10, 100]:
            for lstm_layer in [32, 64, 128]:
                for mdn_layer in [2, 3, 5]:
                    sh_command = command.format(
                        10,
                        batch_size,
                        look_back,
                        lstm_layer,
                        mdn_layer,
                        dancers,
                        tags,
                        look_back)
                    print(sh_command)
                    if call:
                        stream = os.popen(sh_command)
                        output = stream.read()
                        output

run_exp_suite_01()




