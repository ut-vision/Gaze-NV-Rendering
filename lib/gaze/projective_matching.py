import numpy as np



def compute_peri(lm):
	re = np.mean(lm[:2],axis=0)
	le = np.mean(lm[2:4],axis=0)
	me = np.mean(lm[4:6],axis=0)
	return (np.linalg.norm(re-le) + np.linalg.norm(re-me) + np.linalg.norm(me-le))/3.


def parameters(Fc, landmarks, face_model):
	'''
	landmarks : 6 points (4 eye corners + 2 mouth or nose corners)
	face_model : 6 points (4 eye corners + 2 mouth or nose corners)
	'''
	alpha = compute_peri(face_model)/compute_peri(landmarks)
	Fc_depth = np.linalg.norm(Fc,axis=1).reshape(-1,)
	diff = Fc_depth.reshape(-1,1) - alpha * landmarks[:,2].reshape(-1,1)
	beta = np.mean(diff)
	return alpha, beta



def uvd_2_xyz(vertices, camera_matrix):
	d = vertices[:,2]
	u, v = vertices[:,0], vertices[:,1]
	uv1 = np.stack([u,v,np.ones(u.shape)]).T
	# print('uv1 shape: ', uv1.shape)
	inv_cam = np.linalg.inv(camera_matrix)
	temp = uv1 @ inv_cam.T
	temp = temp/np.linalg.norm(temp,axis=1).reshape(-1,1)
	new_vertices = temp * d.reshape(-1,1)    
	return new_vertices
