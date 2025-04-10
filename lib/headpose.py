
import os
import argparse
import math
import numpy as np
import cv2
from scipy.optimize import least_squares
import torch


from utils.read_xgaze import  read_xml,read_csv_as_dict, read_lm_gc



def fun(x,face_model, cameras):
	hr =  np.array([ x[0], x[1], x[2] ]).reshape(3,1); ht = np.array([ x[3], x[4], x[5] ]).reshape(3,1)
	hR = cv2.Rodrigues(hr)[0]
	face_model = face_model[[20, 23, 26, 29, 15, 19], :]
	Points = (np.dot(hR, face_model.T) + ht).T

	proj_lm2d = []
	gt_lm2d = []
	for camera in list(cameras.values()):
		lm_gt, camera_matrix, camera_translation, camera_rotation = camera[0], camera[1], camera[2], camera[3]
		lm_gt = lm_gt[[36, 39, 42, 45, 31, 35], :]
		# lm_gt = lm68_to_50(lm_gt)
		gt_lm2d.append(lm_gt)
		
		fx,fy,cx,cy = camera_matrix[0,0],camera_matrix[1,1], camera_matrix[0,2], camera_matrix[1,2]
		Points_camera =  Points @ camera_rotation.T + camera_translation.reshape(1,3)
		Points_camera = Points_camera/ Points_camera[:,[2]]
		points = Points_camera @ camera_matrix.T
		points = points[:,:2]
		proj_lm2d.append(points)
	proj_lm2d = np.array(proj_lm2d)
	gt_lm2d = np.array(gt_lm2d)

	return proj_lm2d.flatten() - gt_lm2d.flatten()

def get_headpose(face_model, calibration_dir, sub_dict, frame,  init_hr, init_ht):
	cameras = {}
	for i in range(18):
		cam = 'cam' + str(i).zfill(2)
		if cam in [['cam04', 'cam17']]:
			continue
		camera_path = os.path.join(calibration_dir, cam + '.xml')
		camera_matrix, camera_distortion, camera_translation, camera_rotation = read_xml(camera_path)
		lm_gt, gc, _, _ = read_lm_gc(sub_dict, os.path.join(frame,cam+'.JPG'))
		cameras[cam] = [lm_gt, camera_matrix, camera_translation, camera_rotation]

	x0_init = np.concatenate( (init_hr.flatten(), init_ht.flatten() ) )
	# x0_init = np.array([0,0,0,0,0,800])
	solution = least_squares(fun, x0_init, args=(face_model,cameras))
	hr, ht = solution.x[:3].reshape(3,1), solution.x[3:].reshape(3,1)
	return hr,ht
