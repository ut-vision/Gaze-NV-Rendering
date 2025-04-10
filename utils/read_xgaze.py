import csv
import os
import numpy as np
import scipy.io
# from lib.normalize import read_image
import cv2

def read_csv_as_dict(csv_path):
	with open(csv_path, newline='') as csvfile:
		data = csvfile.readlines()
	reader = csv.reader(data)
	sub_dict = {}
	for row in reader:
		frame = row[0]
		cam_index = row[1]
		sub_dict[frame+'/'+cam_index] = row[2:]
	return sub_dict
    
def read_lm_gc(sub_dict, index):
    gaze_point_screen = np.array([int(float(i)) for i in sub_dict[index][0:2]])
    gaze_point_cam = np.array([float(i) for i in sub_dict[index][2:5]])
    head_rotation_cam = np.array([float(i) for i in sub_dict[index][5:8]])
    head_translation_cam = np.array([float(i) for i in sub_dict[index][8:11]])
    lm_2d = np.array([int(float(i)) for i in sub_dict[index][11:]]).reshape(68,2)
    return lm_2d, gaze_point_cam, head_rotation_cam, head_translation_cam


def read_xml(xml_path):
    if not os.path.isfile(xml_path):
        print('no camera calibration file is found.')
        exit(0)
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
    camera_distortion = fs.getNode('Distortion_Coefficients').mat()
    camera_translation = fs.getNode('cam_translation').mat()
    camera_rotation = fs.getNode('cam_rotation').mat()
    return camera_matrix, camera_distortion, camera_translation, camera_rotation

# def read_data(subject, frame, camera_index):
#     dataset_basepath='/home/jqin/Datasets/xgazeraw/data/train'
#     camera_path = os.path.join('/home/jqin/Datasets/xgazeraw/cam_calibration', camera_index.replace('.JPG','.xml'))

#     camera_matrix, camera_distortion, camera_translation, camera_rotation = read_xml(camera_path)
#     # print('camera_matrix: ', camera_matrix)
#     # print('camera_distortion: ', camera_distortion)
#     csv_path = os.path.join('/home/jqin/Datasets/xgazeraw/data/annotation_train', subject+'.csv')

#     img_path = os.path.join(dataset_basepath, subject, frame, camera_index)

#     img = read_image(img_path, camera_matrix, camera_distortion)

#     lm_gt, gc, hr, ht = read_lm_gc(csv_path, os.path.join(frame,camera_index))
    
#     return camera_matrix, camera_distortion,  camera_translation, camera_rotation, img, lm_gt, gc, hr, ht


# def read_data_woimg(subject, frame, camera_index):
#     camera_path = os.path.join('/home/jqin/Datasets/xgazeraw/cam_calibration', camera_index.replace('.JPG','.xml'))

#     camera_matrix, camera_distortion, camera_translation, camera_rotation = read_xml(camera_path)

#     csv_path = os.path.join('/home/jqin/Datasets/xgazeraw/data/annotation_train', subject+'.csv')

#     lm_gt, gc, hr, ht = read_lm_gc(csv_path, os.path.join(frame,camera_index))
    
#     return camera_matrix, camera_distortion,  camera_translation, camera_rotation, lm_gt, gc, hr, ht