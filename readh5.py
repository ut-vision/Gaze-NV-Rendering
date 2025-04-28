import h5py
import imageio
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
from matplotlib import rc
import matplotlib.font_manager
import pandas as pd
import seaborn as sns

sns.set_theme(font_scale=1.5)


font = {'family' : 'sans-serif',
		'serif': 'Helvetica'}
matplotlib.rc('font', **font)



def rad_to_degree(head_pose):
	return head_pose * 180/np.pi


def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
	"""Draw gaze angle on given image with a given eye positions."""
	image_out = image_in
	(h, w) = image_in.shape[:2]
	length = w / 2.0
	pos = (int(h / 2.0), int(w / 2.0))
	if len(image_out.shape) == 2 or image_out.shape[2] == 1:
		image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
	dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
	dy = -length * np.sin(pitchyaw[0])
	cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
				   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
				   thickness, cv2.LINE_AA, tipLength=0.2)
	return image_out



if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str)
	def get_config():
		args, unparsed = parser.parse_known_args()
		return args, unparsed
	args, _ = get_config()

	data_dir = args.data_dir
	sample_dir = os.path.join(data_dir, 'samples')
	os.makedirs(sample_dir, exist_ok=True)

	person_list = sorted(glob.glob(data_dir + '/' + '*.h5'))[:]
	for person_path in person_list[:]:
		with h5py.File(person_path, 'r') as f:
			for key, value in f.items():
				print('number of {}: '.format(key), value.shape[0])

			number_images = f['face_patch'].shape[0]

			p = os.path.basename(person_path).split('.')[0]

			for i in range(  np.minimum(36, number_images) ):
				image = f['face_patch'][i]
				gaze = f['face_gaze'][i]
				head_pose = f['face_head_pose'][i]* (180/np.pi)
				if 'face_mask' in list(f.keys()):
					mask = f['face_mask'][i]
					cv2.imwrite( sample_dir +'/mask_{}_{}_{:.1f}_{:.1f}.jpg'.format( p, i, head_pose[0], head_pose[1]), mask)
	
				image = draw_gaze(image, gaze, thickness=4)
				cv2.imwrite( sample_dir +'/image_{}_{}_{:.1f}_{:.1f}.jpg'.format( p, i, head_pose[0], head_pose[1]), image)
				


