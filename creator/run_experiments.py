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
-S train.epochs={} \
-S train.batch_size={} \
-S train.look_back={} \
-S train.lstm_layer={} \
-S train.mdn_layer={} \
-S generate.look_back={}'

for batch_size in [32, 64, 128]:
    for look_back in [10, 30, 100, 300]:
        for lstm_layer in [32, 64, 128]:
            for mdn_layer in [2, 3, 5]:
                sh_command = command.format(
                    10, batch_size, look_back, lstm_layer, mdn_layer, look_back)
                print(sh_command)
                stream = os.popen(sh_command)
                output = stream.read()
                output




