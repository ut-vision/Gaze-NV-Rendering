import numpy as np
from lib.label_transform import get_rotation, compute_R, rotation_matrix, get_eye_nose_landmarks, get_eye_mouth_landmarks, mean_eye_nose, mean_eye_mouth
from lib.gaze.gaze_utils import vector_to_pitchyaw, pitchyaw_to_vector
from lib.utils.h5_utils import add, to_h5


def np_load(path):
	"""load npy file or txt file based on the extension name"""
	if path.endswith('.npy'):
		return np.load(path)
	elif path.endswith('.txt'):
		return np.loadtxt(path)
	else:
		raise ValueError('unknown file extension: {}'.format(path))



def rotate_mpii(cfg, v_list, source, to_write):

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
		## rotation-to-novel-pose 
		ct = mean_eye_mouth(get_eye_mouth_landmarks(lm_norm)).reshape(1,3)
		lm_new = (lm_norm-ct)@rotation.T + ct
		gaze_new = gaze_norm.reshape((1,3)) @rotation.T
		hR_norm = rotation_matrix( -hr_norm.flatten()[0], hr_norm.flatten()[1], 0)
		hR_new = np.dot(rotation, hR_norm)
		v_new = (v_norm - ct) @ rotation.T + ct

		# normalization again
		R_new = compute_R(get_eye_mouth_landmarks(lm_new), dataname='mpii')
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
		add(to_write, 'rotation_matrix', rotation)

		v_list.append(v_norm_new)

		
