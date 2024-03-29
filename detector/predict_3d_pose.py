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


def run_external_script(input_folder):
    if os.path.exists(input_folder + '/pose2d.npz'):
        print(subprocess.check_output([
            'python', 'run.py',
            '-d', 'custom',
            '-k', input_folder + '/pose2d',
            '-arc', '3,3,3,3,3',
            '-c', 'checkpoint',
            '--evaluate', 'pretrained_243_h36m_detectron_coco_wtraj.bin',
            '--render',
            '--viz-subject', 'detectron2',
            '--viz-action', 'custom',
            '--viz-camera', '0',
            '--viz-export', input_folder + '/pose3d',
            '--viz-size', '6']).decode("utf-8"))
    else:
        print("An error occured. No 2D pose estimation found!")

if __name__ == '__main__':
    print("==========================================================")
    print("to-share::detector")
    print("==========================================================")

    args = get_parser().parse_args()

    os.chdir('VideoPose3D')
    basedir = '../../data/train/video/deep-dance'
    for video_folder in os.listdir(basedir):
        if os.path.isdir(basedir + '/' + video_folder):
            print('\n')
            print('Predicting 3D poses for videos in folder \"' + video_folder + '\"...')
            print("----------------------------------------------------------")
            for folder in os.listdir(basedir + '/' + video_folder):
                input_folder = basedir  + '/' + video_folder + '/' + folder
                pose3d_video = input_folder + '/pose3d.npz'
                print('')
                print('Computing ' + input_folder)
                if path.isfile(pose3d_video):
                    print("3D pose estimation found.")
                    if args.overwrite:
                        print("Flag detected. Overwriting...")
                        run_external_script(input_folder)
                    else:
                        print("Skipping.")
                else:
                    run_external_script(input_folder)

    os.chdir('..')
