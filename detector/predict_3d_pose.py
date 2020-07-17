# This file is part of to-share.

# to-share is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

import os
from os import path
import argparse
import subprocess

def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing 3D estimations on sequence level.")
    return parser

def run_external_script(folder):
    print("----------------------------------------------------------")
    print("to-share::detector")
    print("----------------------------------------------------------")
    print(subprocess.check_output([
        'python3.6', 'run.py',
        '-d', 'custom',
        '-k', 'data/' + folder + '/pose2d',
        '-arc', '3,3,3,3,3',
        '-c', 'checkpoint',
        '--evaluate', 'pretrained_h36m_detectron_coco.bin',
        '--render',
        '--viz-subject', 'detectron2',
        '--viz-action', 'custom',
        '--viz-camera', '0',
        '--viz-video', '../../data/train/' + folder + '/input.mp4',
        '--viz-output', '../../data/train/' + folder + '/pose3d.mp4',
        '--viz-export', '../../data/train/' + folder + '/pose3d',
        '--viz-size', '6']))

if __name__ == '__main__':
    args = get_parser().parse_args()

    os.chdir('VideoPose3D')
    basedir = '../../data/train/'
    for folder in os.listdir(basedir):
        pose3d_video = basedir + folder + '/pose3d.mp4'
        print(pose3d_video)
        if path.isfile(pose3d_video):
            print("3D pose estimation found.")
            if args.overwrite:
                print("Flag detected. Overwriting...")
                run_external_script(folder)
            else:
                print("Skipping.")
        else:
            run_external_script(folder)
        
    os.chdir('..')