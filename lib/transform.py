

import cv2

import numpy as np

INDICES = np.array([36, 39, 42, 45, 31, 35])


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


def lm68_to_50(lm_68):
	'''
	lm_68: (68,2)
	'''
	lm_50 = np.zeros((50,2))
	lm_50[0] = lm_68[8]
	lm_50[1:44] = lm_68[17:60]
	lm_50[44:47] = lm_68[61:64]
	lm_50[47:50] = lm_68[65:68]
	return lm_50

# def compute_R(lm):
# 	'''
# 	lm: lm1 in OpenGL
# 	'''
# 	lm6 = lm[INDICES] # in OpenCV
# 	left_center = np.mean(lm6[2:4,:],axis=0)
# 	right_center = np.mean(lm6[:2,:],axis=0)
# 	center = np.mean(lm6,axis=0)

def compute_R(lm6, dataname):
	'''
	6 landmarks in opencv coordinate
	dataname: mpii or xgaze
	the face center are computed differently 
		for mpii: the 6 landmarks are 4 eye + 2 mouth
		for xgaze: the 6 landmarks are 4 eye + 2 nose
	'''
	if dataname=='mpii':
		left_center = np.mean(lm6[2:4,:],axis=0)
		right_center = np.mean(lm6[:2,:],axis=0)
		face_center = np.mean(lm6,axis=0)
	elif dataname=='xgaze':
		left_center = np.mean(lm6[2:4,:],axis=0)
		right_center = np.mean(lm6[:2,:],axis=0)
		nose_center = np.mean(lm6[[4,5],:],axis=0)
		face_center = ( (left_center + right_center)/2 + nose_center ) /2

	distance = np.linalg.norm(face_center)

	hRx = left_center - right_center
	hRx /= np.linalg.norm(hRx)
	forward = (face_center/distance).reshape(3)
	down = np.cross(forward, hRx)
	down /= np.linalg.norm(down)
	right = np.cross(down, forward)
	right /= np.linalg.norm(right)
	R = np.c_[right, down, forward].T
	return R
	
def rotation_matrix(x, y, z):
	'''
	x, y, z: roll, pitch, yaw, (radians)
	'''
	Rx = np.array([[1,0,0],
				[0, np.cos(x), -np.sin(x)],
				[0, np.sin(x), np.cos(x)]])

	Ry = np.array([[ np.cos(y), 0, np.sin(y)],
				[ 0,         1,         0],
				[-np.sin(y), 0, np.cos(y)]])

	Rz = np.array([[np.cos(z), -np.sin(z), 0],
				[np.sin(z),  np.cos(z), 0],
				[0,0,1]])
	return Rz@Ry@Rx
def get_rotation(from_pose, target_pose):
	
	rotation1 = rotation_matrix( -from_pose[0], from_pose[1], 0)
	rotation2 = rotation_matrix(-target_pose[0], target_pose[1], 0)
	rotation = rotation2@np.linalg.inv(rotation1)
	return rotation

def hR_2_hr(hR):
	def rotation_matrix(x, y, z):
		'''
		x, y, z: pitch, yaw, roll (radians)
		'''
		Rx = np.array([[1,0,0],
					[0, np.cos(x), -np.sin(x)],
					[0, np.sin(x), np.cos(x)]])

		Ry = np.array([[ np.cos(y), 0, np.sin(y)],
					[ 0,         1,         0],
					[-np.sin(y), 0, np.cos(y)]])

		Rz = np.array([[np.cos(z), -np.sin(z), 0],
					[np.sin(z),  np.cos(z), 0],
					[0,0,1]])
		return Rz@Ry@Rx

	hr = np.array([np.arcsin(hR[1, 2]),
				np.arctan2(hR[0, 2], hR[2, 2])])
	return hr

def hr_2_hR(hr):
	hR = rotation_matrix( -hr[0], hr[1], 0)
	return hR