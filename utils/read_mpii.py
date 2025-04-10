import csv
import os
import numpy as np
import scipy.io
# from normalize_data import read_image

def read_txt_as_dict(text_path):
	with open(text_path) as f:
		data = f.readlines()
	reader = csv.reader(data)
	p = {}
	for row in reader:
		words = row[0].split()
		p[words[0]] = words[1:]
	return p
def read_lm_gc(person_dict, index):
	landmarks = np.array([int(i) for i in person_dict[index][2:14]]).reshape((6,2))
	gc = np.array([float(i) for i in person_dict[index][23:26]]).reshape((3,1))
	return landmarks, gc
	
# def read_data(img_name, person, day):
# 	dataset_basepath='/home/jqin/Datasets/MPIIFaceGaze'
# 	camera_path = os.path.join(dataset_basepath,person,'Calibration/Camera.mat')
# 	camera = scipy.io.loadmat(camera_path)
# 	camera_matrix, camera_distortion = camera['cameraMatrix'], camera['distCoeffs']
	
# 	img_path = os.path.join(dataset_basepath, person, day, img_name)
# 	img = read_image(img_path, camera_matrix, camera_distortion)
	
# 	index = os.path.join(day,img_name)
# 	txt_path = os.path.join( os.path.join(dataset_basepath, person), (person+'.txt') )
# 	lm_gt, gc = read_lm_gc(txt_path, index)
# 	return camera_matrix, camera_distortion, img, lm_gt, gc
 