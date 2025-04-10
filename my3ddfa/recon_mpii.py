#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
import glob
import os
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn
from skimage.transform import rescale

import h5py

STD_SIZE = 120

def read_lm_gc(csv_path, index):
    with open(csv_path, newline='') as csvfile:
        data = csvfile.readlines()

    reader = csv.reader(data)
    subject = {}
    for row in reader:
        frame = row[0]
        cam_index = row[1]
        subject[frame+'/'+cam_index] = row[2:]


    gaze_point_screen = [int(float(i)) for i in subject[index][0:2]]
    gaze_point_cam = [float(i) for i in subject[index][2:5]]
    head_rotation_cam = [float(i) for i in subject[index][5:8]]
    head_translation_cam = [float(i) for i in subject[index][8:11]]
    lm_2d = np.array([float(i) for i in subject[index][11:]]).reshape(68,2)
    return lm_2d


def main(args):
    # 1. load pre-tained model
    checkpoint_fp = os.path.join(os.path.dirname(__file__),'models/phase1_wpdc_vdc.pth.tar')
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. load dlib model for face detection and landmark used for face cropping
    if args.dlib_landmark:
        dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if args.dlib_bbox:
        face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    
    
    input_path = args.files
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # # make a specific folder for cropped size obj and lm
    # crop_path = os.path.join(output_path,'obj')
    # if not os.path.exists(crop_path):
    #     os.makedirs(crop_path)


    img_list = glob.glob(input_path + '/' + '*.png')
    img_list += glob.glob(input_path + '/' + '*.jpg')
    img_list += glob.glob(input_path + '/' + '*.JPG')
    img_list += glob.glob(input_path + '/' + '*.PNG')
    sorted(img_list)
    num_images = len(img_list)
    print('this person took {} images this day'.format(num_images))

    for img_fp in img_list: #[:int(num_images/6)]:
        filename = os.path.basename(img_fp)
        out_fp = os.path.join(output_path,filename)
        # crop_fp = os.path.join(crop_path, filename)
        img_ori = cv2.imread(img_fp)
        if args.dlib_bbox:
            max_size = max(img_ori.shape[0], img_ori.shape[1])
            if max_size> 1000:
                print('image shape before rescale: ',img_ori.shape)
                img_ori = rescale(img_ori, 1000./max_size, multichannel=True)
                print('image shape after rescale: ',img_ori.shape)
                img_ori = (img_ori*255).astype(np.uint8)
            rects = face_detector(img_ori, 1)
        else:
            rects = []

        if len(rects) == 0:
            continue

        pts_res = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection, [todo: validate it]
        vertices_lst = []  # store multiple face vertices
        ind = 0
        suffix = get_suffix(img_fp)
        for rect in rects:
            # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
            if args.dlib_landmark:
                # - use landmark for cropping
                pts = face_regressor(img_ori, rect).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                roi_box = parse_roi_box_from_landmark(pts)
            else:
                # - use detected face bbox
                bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                roi_box = parse_roi_box_from_bbox(bbox)

            img = crop_img(img_ori, roi_box)

            # forward: one step
            img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # 68 pts
            pts68, crop_pts, _ = predict_68pts(param, roi_box,transform=True)

            # two-step for more accurate bbox to crop face
            if args.bbox_init == 'two':
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(img_ori, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                pts68 = predict_68pts(param, roi_box,transform=True)

            pts_res.append(pts68)
            P, pose, t = parse_pose(param)
            Ps.append(P)
            poses.append(pose)

            # dense face 3d vertices
            if args.dump_obj: #args.dump_ply or args.dump_vertex or args.dump_depth or args.dump_pncc or
                vertices, crop_vertices, crop_params = predict_dense(param, roi_box,transform=True)
                vertices_lst.append(vertices)
            if args.dump_pts:
                # wfp = '{}_{}.txt'.format(out_fp.replace(suffix, ''), ind)
                wfp = '{}_lm.txt'.format(out_fp.replace(suffix, ''))
                pts68_yx = pts68.copy()
                pts68_yx[2,:] *= -1
                # np.savetxt(wfp, pts68_yx.T, fmt='%.3f')
                # print('Save 68 3d landmarks to {}'.format(wfp))
                crop_pts[2,:] *= -1
                np.savetxt(wfp, crop_pts.T, fmt='%.3f')
                print('Save cropped 68 3d landmarks to {}'.format(wfp))

            if args.dump_obj:
                # wfp = '{}_{}.obj'.format(out_fp.replace(suffix, ''), ind)
                wfp = '{}.obj'.format(out_fp.replace(suffix, ''))
                colors = get_colors(img_ori, vertices)
                # write_obj_with_colors(wfp, vertices, tri, colors)
                # print('Dump obj with sampled texture to {}'.format(wfp))
                write_obj_with_colors(wfp, crop_vertices, tri, colors)
                print('Save cropped obj with sampled texture to {}'.format(wfp))

                pfp = '{}_crop_params.txt'.format(out_fp.replace(suffix, ''))
                np.savetxt(pfp, crop_params, fmt='%.4f')
                print('Save cropping parameters to {}'.format(pfp))

            ind += 1

if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in ('true', '1')
    def add_argument_group(name):
        arg = parser.add_argument_group(name)
        arg_lists.append(arg)
        return arg
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')

    parser.add_argument('--mpii_path', type=str, help='mpii base dir containig the frontal image')
    parser.add_argument('--output_dir', type=str)     

    
    # ------------------------------------------ useless arguments ------------------------------------------ 
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_pts', default='true', type=str2bool)
    parser.add_argument('--dump_obj', default='true', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                            help='whether use dlib landmark to crop image')
    parser.add_argument('--dump_ply', default='false', type=str2bool)
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    # ----------------------------------------------------------------------------------------------------------
    args, unparsed = parser.parse_known_args()

    person_list = sorted(glob.glob(args.mpii_path + '/' + 'p*'))
    for person in person_list[:]:
        day_list = sorted(glob.glob(person + '/' + 'day*'))
        for day_path in day_list[:]:
            print(day_path + ' of {} day'.format(len(day_list)))
            day = day_path.split('/')[-1]
            person = day_path.split('/')[-2]
            # Reconstruction
            args.files = day_path

            obj_path = os.path.join(args.output_dir, person, day)
            os.makedirs(obj_path, exist_ok=True)
            
            args.output_path = obj_path
            main(args)