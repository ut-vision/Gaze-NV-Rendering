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

def myplot(x,y,s,bins=128):
	# rg = np.array([[-80, 80], [-80, 80]])s
	rg = np.array([[-120, 120], [-120, 120]])
	heatmap, xedges, yedges = np.histogram2d(x,y,bins=bins,density=True,range=rg)
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
	return heatmap.T, extent

def plot_heatmap(save_path, to_plot, s):
	x = to_plot[:,1]
	y = to_plot[:,0]
	img, extent = myplot(x,y,s)
	# extent=[-80,80,-80,80]
	# plt.figure()
	plt.clf()
	plt.imshow(img,extent=extent,origin='lower', cmap=cm.jet)
	plt.savefig(save_path)
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

def draw_two_fig(samples, name):
	# plt.figure()
	# plt.scatter(samples[:, 1], samples[:, 0])
	# plt.title("head pose of MPII")
	# plt.xlabel("yaw horizontal (deg)")
	# plt.ylabel('pitch vertical (deg)')
	# plt.savefig(name + '.jpg')
	plot_heatmap(name+'_hm.jpg',samples,16)

def draw_sns(distribution, name):
	plt.figure()
	df_head = pd.DataFrame({"Yaw [degree]": distribution[:,1], "Pitch [degree]":distribution[:,0]})
	h = sns.JointGrid(x="Yaw [degree]", y="Pitch [degree]", data=df_head, xlim=(-150,150), ylim=(-150,150))  
	h.ax_joint.set_aspect('equal')         
	h.plot_joint(sns.histplot)                         
	h.ax_marg_x.set_axis_off()
	h.ax_marg_y.set_axis_off()
	h.ax_joint.set_yticks([-120, -40, 0, 40, 120])
	h.ax_joint.set_xticks([-120, -40, 0, 40, 120])
	plt.savefig(name+'.jpg',bbox_inches='tight')



def plot_distribution(person_list):
	head_pose_all = np.empty((0,2))
	gaze_all =  np.empty((0,2))
	for person_path in person_list:
		with h5py.File(person_path, 'r') as f:
			print('Number of images: ', f['face_patch'].shape[0])
			p = os.path.basename(person_path).split('.')[0]
			
			head_pose_all = np.append(head_pose_all, f['face_head_pose'][:])
			gaze_all = np.append(gaze_all, f['face_gaze'][:])

	head_path = os.path.join(data_dir, 'head_distribution.txt')
	gaze_path = os.path.join(data_dir, 'gaze_distribution.txt')

	np.savetxt(head_path, head_pose_all )
	np.savetxt(gaze_path, gaze_all )

	H_deg = rad_to_degree(head_pose_all)
	G_deg = rad_to_degree(gaze_all)

	draw_sns(H_deg, data_dir+'/H_sns')
	draw_sns(G_deg, data_dir+'/G_sns')
	draw_two_fig(H_deg,data_dir+'/H')
	draw_two_fig(G_deg,data_dir+'/G')

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-plt', '--plot_distribution', action='store_true')
	parser.add_argument('--data_dir', type=str)
	def get_config():
		args, unparsed = parser.parse_known_args()
		return args, unparsed
	args, _ = get_config()

	data_dir = args.data_dir
	sample_dir = os.path.join(data_dir, 'samples')
	os.makedirs(sample_dir, exist_ok=True)

	person_list = sorted(glob.glob(data_dir + '/' + '*.h5'))[:]
	
	if args.plot_distribution:
		plot_distribution(person_list)


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
				


