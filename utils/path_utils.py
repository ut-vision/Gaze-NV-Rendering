import os
import os.path as osp
import numpy as np
import imageio
import cv2
import h5py
import math



def create_paths(args):	
	if args.save_dir is None:
		print('please input the save folder')
		exit(0)
	if args.real_dir is not None:
		os.makedirs(args.real_dir, exist_ok=True)

	os.makedirs(args.save_dir, exist_ok=True)
	################################################################################
	
	syn_dir = osp.join(args.save_dir, 'components/syn')
	cl_dir = osp.join(args.save_dir, 'components/cl') 
	bg_dir = osp.join(args.save_dir, 'components/bg')
	dark_dir = osp.join(args.save_dir, 'components/dark')
	cl_dark_dir = osp.join(args.save_dir, 'components/cl-dark') 
	bg_dark_dir = osp.join(args.save_dir, 'components/bg-dark')

	os.makedirs(syn_dir, exist_ok=True)
	os.makedirs(cl_dir, exist_ok=True)
	os.makedirs(bg_dir, exist_ok=True)
	os.makedirs(dark_dir, exist_ok=True)
	os.makedirs(cl_dark_dir, exist_ok=True)
	os.makedirs(bg_dark_dir, exist_ok=True)
	

	ab1_dir = osp.join(args.save_dir, 'ab1')
	ab2_dir = osp.join(args.save_dir, 'ab2')
	full_dir = osp.join(args.save_dir, 'full')
	runsample_dir = osp.join(args.save_dir, 'run_samples')


	os.makedirs(ab1_dir, exist_ok=True)
	os.makedirs(ab2_dir, exist_ok=True)
	os.makedirs(full_dir, exist_ok=True)
	os.makedirs(runsample_dir, exist_ok=True)

	args.syn_dir = syn_dir
	args.cl_dir = cl_dir
	args.bg_dir = bg_dir
	args.dark_dir = dark_dir
	args.cl_dark_dir = cl_dark_dir
	args.bg_dark_dir = bg_dark_dir
	args.ab1_dir = ab1_dir
	args.ab2_dir = ab2_dir
	args.full_dir = full_dir
	args.runsample_dir = runsample_dir
	return args

