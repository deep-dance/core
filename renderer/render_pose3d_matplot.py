import warnings
warnings.filterwarnings('ignore')

import os
import subprocess
import tempfile
import shutil
import argparse
import glob

from pathlib import Path

import time
import math
import numpy as np

import common

def plot_frames(fig, ax, keypoints, max_frames, path):
    common.notice("Writing sequence file(s) to \"" + path + "\".")
    for i in range(0, max_frames):
        common.plot_pose(ax, keypoints[i])
        image_path = path + '/seq_001_frame_' + str(i) + '.png'
        fig.savefig(image_path, bbox_inches='tight')
        common.progress("Rendering frame", i)
    print('')

def create_video():
    common.notice("Creating video from image sequence.")
    subprocess.check_output([
        'ffmpeg',
        '-i', tmp + '/seq_001_frame_%d.png',
        '-r', '30',
        '-b:v', '2M',
        'pose3d.avi'])

def get_parser():
    parser = argparse.ArgumentParser(description="Pose3D numpy archive renderer.")
    parser.add_argument("input", help="Path to .npz file.")
    parser.add_argument("--output", help="Output path of rendered video.")
    parser.add_argument("--frames", type=int, help="The number of frames to render.")
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.input:
        input_path = Path(args.input)
        assert input_path.exists(), "The input path(s) was not found."

    common.notice("Loading \"" + str(input_path) + "\".")
    keypoints = np.load(input_path, allow_pickle=True)
    keypoints = keypoints['arr_0']

    max_frames = len(keypoints)
    if args.frames:
        max_frames = args.frames

    assert max_frames <= len(keypoints), "Target frame count exceeds input frame count."

    tmp = tempfile.mkdtemp()

    fig, ax = common.init_plot()
    plot_frames(fig, ax, keypoints, max_frames, tmp)
    create_video()
    
    shutil.rmtree(tmp)

    