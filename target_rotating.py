import numbers
import os
import os.path as osp
import argparse
import math
import numpy as np
import scipy.io
import glob
import cv2
import h5py
import time
import torch
import torch.nn as nn

from lib.transform import get_rotation, compute_R, rotation_matrix, hR_2_hr, hr_2_hR, lm68_to_50
from utils import load_color, write_obj_with_colors, pitchyaw_to_vector, vector_to_pitchyaw, angular_error, draw_gaze, to_h5, add, read_resize_blur

def get_face_center(landmarks_3d):
	'''
	landmarks_3d: (3, 6)
	--> 
	face_center: (3,1)
	'''
	two_eye_center = np.mean(landmarks_3d[:, 0:4], axis=1).reshape((3, 1))
	nose_center = np.mean(landmarks_3d[:, 4:6], axis=1).reshape((3, 1))
	face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))
	return face_center

def np_load(path):
	"""load npy file or txt file based on the extension name"""
	if path.endswith('.npy'):
		return np.load(path)
	elif path.endswith('.txt'):
		return np.loadtxt(path)
	else:
		raise ValueError('unknown file extension: {}'.format(path))



# # rotate ===========================================================
def rotate(cfg, dataname,  v_list, source, to_write):

	target_mode:str = cfg.mode 
	number_new_views:int = cfg.number_new_views 
	
	v_norm = source[0]
	hr_norm = source[1]
	lm_norm = source[2]
	gaze_norm = source[3]
	gaze_norm_py = vector_to_pitchyaw(-gaze_norm.reshape((1,3))).flatten()

	good_case = 0
	#  xgaze-train: (42030, 18, 2)
	#  EYEDIAP CS  # (19221,2)
	#  EYEDIAP FT  # (22567,2)

	while (good_case < number_new_views):
		if target_mode in ['xgaze-train', 'eyediap-cs', 'eyediap-ft']:
			headpose_file = cfg.headpose_file
			## sample from supplementary target head pose distribution
			large_poses = np_load(headpose_file).reshape(-1,2)
			sample_idx = np.random.randint(large_poses.shape[0])		
			pose = large_poses[sample_idx] 
			# compute the rotation matrix to the target head pose
			rotation = get_rotation(hr_norm.flatten(), pose.flatten())


		elif target_mode in ['gaussian']:
			config_sigma:float = cfg.sigma
			sample_target_gaze:bool = cfg.sample_target_gaze

			mu, sigma = 0*(np.pi/180), float(config_sigma) * (np.pi/180)  # mean and standard deviation
			if sample_target_gaze:
				target_gaze = np.random.normal(mu, sigma, size=(2,))
				# print('target_gaze: ', target_gaze)
				rotation = get_rotation(gaze_norm_py.flatten(), target_gaze.flatten())
			else:
				pose = np.random.normal(mu, sigma, size=(2,))
				rotation = get_rotation(hr_norm.flatten(), pose.flatten())
		
		
		###############################################################################
		## rotation-to-novel-pose starts here
		if dataname == 'mpii':
			ct = np.mean(lm_norm[[36,39,42,45,48,54],:],axis=0).reshape(1,3)
			lm_new = (lm_norm-ct)@rotation.T + ct
			gaze_new = gaze_norm.reshape((1,3)) @rotation.T
			hR_norm = rotation_matrix( -hr_norm.flatten()[0], hr_norm.flatten()[1], 0)
			hR_new = np.dot(rotation, hR_norm)
			v_new = (v_norm - ct) @ rotation.T + ct

			# normalization again
			R_new = compute_R(lm_new[[36,39,42,45,48,54]], dataname=dataname)
			lm_norm_new = (lm_new-ct) @ R_new.T + ct
			gaze_norm_new = gaze_new.reshape((1,3)) @ R_new.T
			hR_norm_new = R_new @ hR_new
			hr_norm_new = np.array([np.arcsin(hR_norm_new[1, 2]),
					np.arctan2(hR_norm_new[0, 2], hR_norm_new[2, 2])])

			if np.linalg.norm(hr_norm_new)* (180/np.pi) > 80:
				print(f'head pose = {hr_norm_new* (180/np.pi)} is too large')
				continue
			
			v_norm_new = (v_new - ct) @ R_new.T + ct
			good_case += 1

			add(to_write, 'face_gaze', vector_to_pitchyaw(-gaze_norm_new).flatten())
			add(to_write, 'face_head_pose', hr_norm_new)
			# add(to_write, 'face_mat_norm', np.transpose(R_new).astype(np.float32),)
			add(to_write, 'rotation_matrix', rotation)

			v_list.append(v_norm_new)

			# target_poses.append(hr_norm_new)


		elif dataname =='xgaze':
			ct = get_face_center( lm_norm[[36,39,42,45,31,35],:].T ).reshape(1,3)
			lm_new = (lm_norm-ct)@rotation.T + ct
			gaze_new = gaze_norm.reshape((1,3)) @rotation.T
			hR_norm = rotation_matrix( -hr_norm.flatten()[0], hr_norm.flatten()[1], 0)
			hR_new = np.dot(rotation, hR_norm)

			gaze_norm_new = gaze_new.copy()
			lm_norm_new = lm_new.copy()
			hR_norm_new = hR_new.copy()
			hr_norm_new = np.array([np.arcsin(hR_norm_new[1, 2]),
					np.arctan2(hR_norm_new[0, 2], hR_norm_new[2, 2])])

			if np.linalg.norm(hr_norm_new)* (180/np.pi) > 80:
				print(f'head pose = {hr_norm_new* (180/np.pi)} is too large')
				continue
			#################################################################################
			## Found good case, add to result list
			v_new = (v_norm-ct)@rotation.T + ct
			good_case += 1

			add(to_write, 'face_gaze', vector_to_pitchyaw(-gaze_norm_new).flatten())
			add(to_write, 'face_head_pose', hr_norm_new)
			# add(to_write, 'face_mat_norm', np.transpose(R_new).astype(np.float32),)
			add(to_write, 'rotation_matrix', rotation)
			v_list.append(v_new)
			
		else:
			print('indicate which source data is used in target_rotating.py')

			


		



if __name__ == '__main__':
	current_dir = osp.dirname(osp.realpath(__file__))
	import argparse
	parser = argparse.ArgumentParser()
	def get_config():
		config, unparsed = parser.parse_known_args()
		return config, unparsed
	config, _ = get_config()

