import os
import os.path as osp
import argparse
import math
import numpy as np
import imageio
import scipy.io
import glob
import cv2
import matplotlib.pyplot as plt
import h5py
import time
from tqdm import tqdm
from omegaconf import OmegaConf
from scipy.optimize import least_squares
import matplotlib.path as mplPath
import torch
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings
from pytorch3d.renderer import (
	PerspectiveCameras, 
	FoVPerspectiveCameras,
	PointLights, 
	DirectionalLights, 
	Materials, 
	# RasterizationSettings, 
	MeshRenderer, 
	MeshRasterizer,  
	SoftPhongShader,
	TexturesUV,
	TexturesVertex,
)
import torch.nn as nn

from lib.normalize import read_image, normalize, normalize_face, load_facemodel, estimateHeadPose, normalize_woimg, resize_landmarks
from lib.projective_matching import parameters, compute_peri, uvd_2_xyz
from lib.headpose import get_headpose
from lib.transform import get_rotation, compute_R, rotation_matrix, hR_2_hr, hr_2_hR, lm68_to_50
from lib.myPytorch3d import SimpleShader, hard_rgb_blend_with_background, BlendParams, mySoftPhongShader

from utils import load_color, write_obj_with_colors, pitchyaw_to_vector, vector_to_pitchyaw, angular_error, draw_gaze, to_h5, add, read_resize_blur
from utils.read_mpii import read_txt_as_dict, read_lm_gc
from set_renderer import renderer1, renderer2, focal_norm, distance_norm, roi_size, run_render
from target_rotating import rotate




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def get_mask_index(lm_gt, vertices, lm_3d, crop_params):
	'''
	lm_gt: ground truth 2D landmarks
	vertices: 3D face vertices from reconstruction
	lm_3d: 3D landmarks from reconstruction
	crop_params: cropping matrix from reconstruction
	'''
	lm_temp = np.c_[ lm_gt, np.ones(68) ] 
	cropped_landmarks =  np.matmul(lm_temp, np.transpose(crop_params))
	outline = cropped_landmarks[[*range(17), *range(26,16,-1)],:2]
	bbPath = mplPath.Path(outline)
	face_mask = bbPath.contains_points(vertices[:,:2]) 
	lm_jaw = lm_3d[5,2] # the 5th index is the landmark near the jaw of face, only want the closer part than it
	jaw_mask = np.where(vertices[:,2]<lm_jaw, True, False)
	new_mask = face_mask * jaw_mask # (n_vertices,)
	mask_index = np.where(new_mask==True)[0]
	return mask_index


def get_face_center(landmarks_3d):
	''' landmarks_3d: (3, 6) -> face_center: (3,1)'''
	face_center = np.mean(landmarks_3d, axis=1).reshape((3, 1))
	return face_center



def main( i ):
	global SOURCE_INDEX

	if (SOURCE_INDEX + 1 ) % 150 == 0:
		print(f"The SOURCE_INDEX: {SOURCE_INDEX} / {total_source_image}" )

	random_color = random_colors[SOURCE_INDEX]
	bg_img_idx = random_bg_ids[SOURCE_INDEX]
	bg_img_paths = sorted(glob.glob(data_cfg.background_path + '/' + '*.jpg'))
	bg_img = read_resize_blur(bg_img_paths[ bg_img_idx ], roi_size)
	ac = random_lightings_ac[SOURCE_INDEX]
	dc = 0.0
	sc = 0
	light = DirectionalLights(
		device=device, 
		ambient_color=((ac, ac, ac), ), 
		diffuse_color=((dc, dc, dc), ), 
		specular_color=((sc, sc, sc),), 
		direction=[[0.0, 0.0, 1.0]]
		)

	which_ids_16 = which_ids[SOURCE_INDEX]
	SOURCE_INDEX += 1
	
	################################# start main ###############################################

	file_name = f['file_name'][i].decode('utf-8') #p00/day01/0005.jpg
	gc = f['3d_gaze_target'][i]

	person = file_name.split('/')[0]
	day = file_name.split('/')[1]
	img_name =  file_name.split('/')[2]
	proper_case = 0


	obj_path = osp.join(obj_dir, person, day, img_name.replace('.jpg', '.obj'))

	if not osp.isfile(obj_path):
		print('obj path does not exist, skip')
		return  
	lm = np.loadtxt(obj_path.replace('.obj', '_lm.txt'))
	crop_params = np.loadtxt(obj_path.replace('.obj', '_crop_params.txt'))

	############################# Load obj ############################       
	c = load_color(obj_path)
	color = torch.tensor(c).to(device)
	texture = TexturesVertex(verts_features=torch.unsqueeze(color, 0))
	vert, face, aux = load_obj(obj_path, device='cpu')
	v = vert.data.numpy()

	############################# Load dataset ############################

	txt_path = osp.join( osp.join(data_dir, person), (person+'.txt') )
	person_dict = read_txt_as_dict(txt_path)
	camera_path = osp.join(data_dir, person,'Calibration/Camera.mat')
	camera = scipy.io.loadmat(camera_path)
	camera_matrix, camera_distortion = camera['cameraMatrix'], camera['distCoeffs']
	img_path = osp.join(data_dir, person, day, img_name)
	img = read_image(img_path, camera_matrix, camera_distortion)

	# load "ground truth 2D-landmarks" and "gaze target" from dataset
	lm_gt, gc = read_lm_gc(person_dict, osp.join(day,img_name))

	############################ Data Normalization ############################

	### estimate head pose 
	face_model = load_facemodel(misc_cfg.mpii_face_model)# (3,6)
	facePts = face_model.T.reshape(6, 1, 3)
	landmarks_sub = lm_gt.astype(np.float32)  # input to solvePnP function must be float type
	landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
	hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion, iterate=True)
	## compute estimated 3D positions of the landmarks
	ht = ht.reshape((3,1))
	hR = cv2.Rodrigues(hr)[0] # rotation matrix
	Fc = np.dot(hR, face_model) + ht # (3,6)
	face_center = np.mean(Fc, axis=1).reshape((3, 1))
	gaze = gc.reshape(1,3) - face_center.reshape(1,3)

	############################# normalize image ############################
	norm_list = normalize(img, lm_gt, focal_norm, distance_norm, roi_size, face_center, hr, ht, camera_matrix, gc)
	img_face, R, hR_norm, gaze_norm, lm_gt_norm = norm_list[0], norm_list[1], norm_list[2], norm_list[3], norm_list[4]
	hr_norm = np.array([np.arcsin(hR_norm[1, 2]), np.arctan2(hR_norm[0, 2], hR_norm[2, 2])])

	for x,y in lm_gt_norm:
		cv2.circle(img_face, (int(x),int(y)), radius=5, color=(255, 0, 0), thickness=5)
		
	#==================================================== Get Mask Index ====================================================================
	lm68 = f['landmarks_2d_detect'][i]
	if lm68 is None:
		return
	mask_index = get_mask_index(lm68, v, lm, crop_params)
	if mask_index.shape[0] == 0:
		return
	proper_case += 1

	# masked_textures
	mask_color = torch.zeros_like(color,device=device)
	mask_color[mask_index] = color[mask_index]
	mask_texture = TexturesVertex(verts_features=torch.unsqueeze(mask_color, 0))

	''' Projective matching: transform the face vertices and 3D landmarks to physical space
		v: face vertices
		lm: 3D landmarks
	''' 
	alpha, beta = parameters(Fc.T,lm[[36,39,42,45,48,54]],openface[[36,39,42,45,48,54]])
	v = v * alpha + np.array([0,0,beta]) 
	lm = lm * alpha + np.array([0,0,beta])
	crop_params *= alpha; crop_params[2,2] = 1
	lm = uvd_2_xyz(lm, crop_params@camera_matrix)
	v = uvd_2_xyz(v, crop_params@camera_matrix)
	
	# better to use 68 landmarks to estimate head pose since only 6 landmarks is not stable or accurate
	def headpose_68(lm68):
		landmarks_68 = lm68.astype(np.float32).reshape(68,1,2)
		facePts = openface.reshape(68,1,3)
		hr, ht = estimateHeadPose(landmarks_68, facePts, camera_matrix, camera_distortion, iterate=True)
		norm_list = normalize_woimg(lm_gt, focal_norm, distance_norm, roi_size, face_center, hr, ht, camera_matrix, gc)
		_, R, hR_norm, _ = norm_list[0], norm_list[1], norm_list[2], norm_list[3]
		hR = cv2.Rodrigues(hr)[0]
	
		hR_norm = np.dot(R, hR)
		hr_norm = np.array([np.arcsin(hR_norm[1, 2]),
					np.arctan2(hR_norm[0, 2], hR_norm[2, 2])])
		return hr_norm
	hr_norm = headpose_68(lm68)

	################################## Rendering #######################################
	
	### rotate and move the face mesh to the normalized position (not moving camera)
	v_norm = v @ R.T
	lm_norm = lm @ R.T
	# move to the normalized distance
	temp = get_face_center(lm_norm[[36, 39, 42, 45, 31, 35],:].T).reshape(1,3) # input shape should be (3, 6)
	cam_move = temp * (distance_norm/np.linalg.norm(temp)) - temp
	v_norm += cam_move.reshape(1,3)
	lm_norm += cam_move.reshape(1,3)

	v_list = []
	to_write = {}
	real_to_write = {}

	save_original_headpose = model_cfg.save_original_headpose
	if save_original_headpose:
		print('save original: ', model_cfg.save_original_headpose)
		v_list.append(v_norm)
		add(to_write, 'landmarks_norm', resize_landmarks(lm_gt_norm, focal_norm, roi_size))
		add(to_write, 'face_gaze', vector_to_pitchyaw(-gaze_norm.reshape((1,3))).flatten())
		add(to_write, 'face_head_pose', hr_norm.astype(np.float32))
		add(to_write, 'face_mat_norm', R.astype(np.float32))
		add(to_write, 'rotation_matrix', np.eye(3)) # relative rotation matrix to source image


	''' rotate: rotate face and gaze to other cameras , put the new labels into to_write:Dict
		after rotate, the v_list will contain extra rotated face vertices
	'''
	source = [v_norm, hr_norm, lm_norm, gaze_norm]
	rotate( model_cfg.target_pose,
			dataname='mpii', 
			v_list=v_list, 
			source=source, 
			to_write=to_write)
	
	## ======================== Rendering =============================
	verts = [torch.tensor(v, dtype=torch.float32,device=device) * torch.tensor([1,-1,-1],device=device) for v in v_list]
	faces = [face.verts_idx.to(device) for v in verts]
	textures = texture.extend(len(verts))
	mesh = Meshes(verts=verts, faces=faces, textures=textures)

	## do the rendering
	images0,images,images_bg,images_dark,images_cl_dark,images_bg_dark = run_render(mesh, random_color, bg_img)

	# get masked image		
	mask_color = torch.zeros_like(color,device=device)
	mask_color[mask_index] = color[mask_index]
	mask_texture = TexturesVertex(verts_features=torch.unsqueeze(mask_color, 0))
	mask_textures = mask_texture.extend(len(verts))
	mask_mesh = Meshes(verts=verts, faces=faces, textures=mask_textures)
	### THIS IS IMPORTANT #######################
	renderer1.shader.blend_params =  BlendParams()
	#############################################
	images_masked = renderer1(mask_mesh,zfar=2000)
	images_masked = images_masked.cpu().data.numpy()[:,:,:,:3]

	## get mask and saving
	for i in range(images0.shape[0]):
		# get binary mask
		temp = cv2.cvtColor(images_masked[i], cv2.COLOR_RGB2BGR)
		binary_mask = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
		ret, binary_mask = cv2.threshold(binary_mask, 2, 255, cv2.THRESH_BINARY)
		face_mask224 = cv2.resize(binary_mask,(224,224), interpolation=cv2.INTER_AREA).astype(np.uint8)
		kernel = np.ones((5,5), np.uint8)
		face_mask224 =  cv2.dilate(face_mask224, kernel, iterations = 3)
		face_mask224 = cv2.erode(face_mask224, kernel, iterations = 3)
		add(to_write, 'face_mask', face_mask224)
		if args.real_dir is not None:
			add(real_to_write, 'face_mask', face_mask224)
	

	to_write_full = to_write.copy()

	for i in range(images0.shape[0]):
		# --------------------------this is for original black bg output---------------------------
		images0[i] = cv2.cvtColor(images0[i], cv2.COLOR_RGB2BGR)
		image0 = cv2.resize(images0[i],(224,224), interpolation=cv2.INTER_AREA).astype(np.uint8)
		# --------------------------this is for random color output---------------------------
		images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
		image_cl = cv2.resize(images[i],(224,224), interpolation=cv2.INTER_AREA).astype(np.uint8)
		# --------------------------this is for bg img output---------------------------
		images_bg[i] = cv2.cvtColor(images_bg[i], cv2.COLOR_RGB2BGR)
		image_bg = cv2.resize(images_bg[i],(224,224), interpolation=cv2.INTER_AREA).astype(np.uint8)
		# --------------------------this is for darker output---------------------------
		images_dark[i] = cv2.cvtColor(images_dark[i], cv2.COLOR_RGB2BGR)
		image_dark = cv2.resize(images_dark[i],(224,224), interpolation=cv2.INTER_AREA).astype(np.uint8)
		# --------------------------this is for darker random color output---------------------------
		images_cl_dark[i] = cv2.cvtColor(images_cl_dark[i], cv2.COLOR_RGB2BGR)
		image_cl_dark = cv2.resize(images_cl_dark[i],(224,224), interpolation=cv2.INTER_AREA).astype(np.uint8)
		# --------------------------this is for darker background img output---------------------------
		images_bg_dark[i] = cv2.cvtColor(images_bg_dark[i], cv2.COLOR_RGB2BGR)
		image_bg_dark = cv2.resize(images_bg_dark[i],(224,224), interpolation=cv2.INTER_AREA).astype(np.uint8)

		add(to_write, 'face_patch', image0)

		aug_idx = which_ids_16[i]

		## full 
		if aug_idx < 0.1:
			image_combine = image0.copy()
		elif aug_idx >= 0.1 and aug_idx < 0.2:
			image_combine = image_dark.copy()
		elif aug_idx >= 0.2 and aug_idx < 0.3:
			image_combine = image_cl.copy()
		elif aug_idx >= 0.3 and aug_idx < 0.4:
			image_combine = image_cl_dark.copy()
		elif aug_idx >= 0.4 and aug_idx < 0.7:
			image_combine = image_bg.copy()
		else:
			image_combine = image_bg_dark.copy()
		add(to_write_full, 'face_patch', image_combine)#

	"""dict to be written
		to_write_full: the final version of the dataset
		real_to_write: simply the normalized image of the cam00.JPG, basically no use
	"""

	to_h5(to_write_full, osp.join(args.full_dir, person + '.h5') )
	if args.real_dir is not None:
		to_h5(real_to_write,  osp.join(args.real_dir, person + '.h5') )




if __name__ == '__main__':

	from utils.path_utils import create_paths

	arg_lists = []
	parser = argparse.ArgumentParser(description='RAM')


	def str2bool(v):
		return v.lower() in ('true', '1')
		

	parser = argparse.ArgumentParser()
	## xgaze
	parser.add_argument('-save', '--save_dir', type=str)
	parser.add_argument('-real', '--real_dir', type=str, default=None)
	parser.add_argument('-ablation', '--ablation', type=bool, default=False)
	parser.add_argument('--group', type=int, help='there are 4 groups of subjects, which group to use (just for parallel rendering)')

	args, unparsed = parser.parse_known_args()


	if args.save_dir is None:
		print('please input the save folder')
		exit(0)
	args.full_dir = osp.join(args.save_dir, 'full')
	os.makedirs(args.full_dir, exist_ok=True)

	## load config
	cfg = OmegaConf.load('./configs/config.yaml')
	data_cfg = cfg.data
	supp_cfg = cfg.supplementary.mpii
	model_cfg = cfg.model
	misc_cfg = cfg.misc

	## set dataset path
	data_dir = data_cfg.mpii.raw
	obj_dir = data_cfg.mpii.obj

	## misc and supplementary
	total_source_image = 22500
	openface = np.loadtxt(misc_cfg.openface_path)
	'''to preserve reproducibility, the random settings of background, color, lighting are all saved in supplementary'''
	## load random index for lighting,  background color, background image
	whether_dark = np.loadtxt(supp_cfg.whether_dark)
	random_lightings_ac = np.loadtxt(supp_cfg.random_lightings_ac)
	random_colors = np.loadtxt(supp_cfg.random_colors).astype(int)
	random_bg_ids = np.loadtxt(supp_cfg.random_bg_ids).astype(int)
	which_ids = np.loadtxt(supp_cfg.which_ids)
	
	SOURCE_INDEX = 0

	## Load the image based on the supplementary files 
	"""  the supplementary (e.g.: ./supplementary/mpii/source_supply/p00.h5) contains the index to the source image that is after filtering
	keys:
		'file_name': e.g.: "p13/day01/0006.jpg"
		'3d_gaze_target'
		'camera_parameters'
		'distortion_parameters'
		'head_pose'
		'landmarks_2d_detect'
	"""
	person_list = sorted(glob.glob(supp_cfg.source + '/' + 'p*.h5'))

	for person_path in person_list[:]:
		start_time = time.time()
		with h5py.File(person_path, 'r') as f:
			print('For example, the first file name: ', f['file_name'][0].decode('utf-8'))
			num =  f['file_name'].shape[0]
			print('num of entries: ', num)

			previous_time = time.time()
			for i in tqdm(range( num )):
				main(i)

		print('one person time: ', time.time()-start_time)

