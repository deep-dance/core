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
import shutil
import argparse
import numpy as np
import cv2
import subprocess as sp
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def get_parser():
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing 2D estimations on sequence level.")
	return parser


def get_img_paths(imgs_dir):
	img_paths = []
	for dirpath, dirnames, filenames in os.walk(imgs_dir):
		for filename in [f for f in filenames if f.endswith('.png') or f.endswith('.PNG') or f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg') or f.endswith('.JPEG')]:
			img_paths.append(os.path.join(dirpath,filename))
	img_paths.sort()

	return img_paths

def read_images(dir_path):
	img_paths = get_img_paths(dir_path)
	for path in img_paths:
		yield cv2.imread(path)


def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    for line in pipe.stdout:
        w, h = line.decode().strip().split(',')
        return int(w), int(h)


def read_video(filename):
    w, h = get_resolution(filename)

    command = ['ffmpeg',
            '-i', filename,
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vsync', '0',
            '-vcodec', 'rawvideo', '-']

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    while True:
        data = pipe.stdout.read(w*h*3)
        if not data:
            break
        yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))


def init_pose_predictor(config_path, weights_path, cuda=True):
	cfg = get_cfg()
	cfg.merge_from_file(config_path)
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	cfg.MODEL.WEIGHTS = weights_path
	if cuda == False:
		cfg.MODEL.DEVICE='cpu'
	predictor = DefaultPredictor(cfg)

	return predictor


def encode_for_videpose3d(boxes,keypoints,resolution, dataset_name):
	# Generate metadata:
	metadata = {}
	metadata['layout_name'] = 'coco'
	metadata['num_joints'] = 17
	metadata['keypoints_symmetry'] = [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]
	metadata['video_metadata'] = {dataset_name: resolution}

	prepared_boxes = []
	prepared_keypoints = []
	for i in range(len(boxes)):
		if len(boxes[i]) == 0 or len(keypoints[i]) == 0:
			# No bbox/keypoints detected for this frame -> will be interpolated
			prepared_boxes.append(np.full(4, np.nan, dtype=np.float32)) # 4 bounding box coordinates
			prepared_keypoints.append(np.full((17, 4), np.nan, dtype=np.float32)) # 17 COCO keypoints
			continue

		prepared_boxes.append(boxes[i])
		prepared_keypoints.append(keypoints[i][:,:2])
		
	boxes = np.array(prepared_boxes, dtype=np.float32)
	keypoints = np.array(prepared_keypoints, dtype=np.float32)
	keypoints = keypoints[:, :, :2] # Extract (x, y)
	
	# Fix missing bboxes/keypoints by linear interpolation
	mask = ~np.isnan(boxes[:, 0])
	indices = np.arange(len(boxes))
	for i in range(4):
		boxes[:, i] = np.interp(indices, indices[mask], boxes[mask, i])
	for i in range(17):
		for j in range(2):
			keypoints[:, i, j] = np.interp(indices, indices[mask], keypoints[mask, i, j])
	
	print('{} total frames processed'.format(len(boxes)))
	print('{} frames were interpolated'.format(np.sum(~mask)))
	print('----------')
	
	return [{
		'start_frame': 0, # Inclusive
		'end_frame': len(keypoints), # Exclusive
		'bounding_boxes': boxes,
		'keypoints': keypoints,
	}], metadata


def predict_pose(pose_predictor, img_generator, output_path, dataset_name='detectron2'):
	'''
		pose_predictor: The detectron's pose predictor
		img_generator:  Images source
		output_path:    The path where the result will be saved in .npz format
	'''
	boxes = []
	keypoints = []
	resolution = None

	# Predict poses:
	for i, img in enumerate(img_generator):
		pose_output = pose_predictor(img)

		if len(pose_output["instances"].pred_boxes.tensor) > 0:
			cls_boxes = pose_output["instances"].pred_boxes.tensor[0].cpu().numpy()
			cls_keyps = pose_output["instances"].pred_keypoints[0].cpu().numpy()
		else:
			cls_boxes = np.full((4,), np.nan, dtype=np.float32)
			cls_keyps = np.full((17,3), np.nan, dtype=np.float32)   # nan for images that do not contain human

		boxes.append(cls_boxes)
		keypoints.append(cls_keyps)

		# Set metadata:
		if resolution is None:
			resolution = {
				'w': img.shape[1],
				'h': img.shape[0],
			}

		print('{}      '.format(i+1), end='\r')

	# Encode data in VidePose3d format and save it as a compressed numpy (.npz):
	data, metadata = encode_for_videpose3d(boxes, keypoints, resolution, dataset_name)
	output = {}
	output[dataset_name] = {}
	output[dataset_name]['custom'] = [data[0]['keypoints'].astype('float32')]
	np.savez_compressed(output_path, positions_2d=output, metadata=metadata)

	print ('All done!')

def run_internal_script(folder):
	# Predict poses and save the result:
	print("----------------------------------------------------------")
	print("to-share::detector")
	print("----------------------------------------------------------")
	img_generator = read_video('../data/train/' + folder + '/input.mp4')  # or get them from a video
	output_path = '../data/train/' + folder + '/pose2d'
	predict_pose(pose_predictor, img_generator, output_path)



if __name__ == '__main__':
	# Init pose predictor:
	model_config_path = 'detectron2/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml'
	model_weights_path = 'detectron2://COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x/139686956/model_final_5ad38f.pkl'

	pose_predictor = init_pose_predictor(model_config_path, model_weights_path, cuda=True)

	args = get_parser().parse_args()

	basedir = '../data/train/'
	for folder in os.listdir(basedir):
		pose2d_numpy = basedir + folder + '/pose2d.npz'
		if path.isfile(pose2d_numpy):
			print("2D pose estimation export (numpy) found.")
			if args.overwrite:
				print("Flag detected. Overwriting...")
				run_internal_script(folder)
			else:
				print("Skipping.")
		else:
			run_internal_script(folder)
		
		# Copy 2D pose numpy files to VideoPose3D data directory, since their scripts do not work otherwise
		videopose3d_path = 'VideoPose3D/data/data_2d_custom_data/' + folder

		os.makedirs(videopose3d_path, exist_ok=True)
		shutil.copyfile(pose2d_numpy, videopose3d_path + '/pose2d.npz')

	




	
