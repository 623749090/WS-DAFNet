from __future__ import print_function
import os
import cv2
import numpy as np
import math
import scipy.io as sio
import re
import dlib
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='WS-DAFNet')
    parser.add_argument('--image_path', default='../dataset/BP4D/', type=str, help='original image path')
    args = parser.parse_args()
    return args
args = parse_args()

reflect_66 = sio.loadmat('data/reflect_66.mat')
reflect_66 = reflect_66['reflect_66']
reflect_66 = reflect_66.reshape(reflect_66.shape[1])
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

crop_name = 'BP4D_crop'
if not os.path.exists('../dataset/' + crop_name):
    os.mkdir('../dataset/' + crop_name)
name_file = open('../source/BP4D_crop_name.txt', 'w')
shape_file = open('../source/BP4D_crop_shape.txt', 'w')

box_enlarge = 3
img_size = 238


def align_face_49pts(img, np_shape, box_enlarge, img_size):

    img_land = np.zeros(len(reflect_66) * 2)
    img_land[0:img_land.shape[0]:2] = np_shape[2 * reflect_66 - 2]
    img_land[1:img_land.shape[0]:2] = np_shape[2 * reflect_66 - 1]

    leftEye0 = (img_land[2 * 19] + img_land[2 * 20] + img_land[2 * 21] + img_land[2 * 22] + img_land[2 * 23] +
                img_land[2 * 24]) / 6.0
    leftEye1 = (img_land[2 * 19 + 1] + img_land[2 * 20 + 1] + img_land[2 * 21 + 1] + img_land[2 * 22 + 1] +
                img_land[2 * 23 + 1] + img_land[2 * 24 + 1]) / 6.0
    rightEye0 = (img_land[2 * 25] + img_land[2 * 26] + img_land[2 * 27] + img_land[2 * 28] + img_land[2 * 29] +
                 img_land[2 * 30]) / 6.0
    rightEye1 = (img_land[2 * 25 + 1] + img_land[2 * 26 + 1] + img_land[2 * 27 + 1] + img_land[2 * 28 + 1] +
                 img_land[2 * 29 + 1] + img_land[2 * 30 + 1]) / 6.0
    deltaX = (rightEye0 - leftEye0)
    deltaY = (rightEye1 - leftEye1)
    l = math.sqrt(deltaX * deltaX + deltaY * deltaY)
    sinVal = deltaY / l
    cosVal = deltaX / l
    mat1 = np.mat([[cosVal, sinVal, 0], [-sinVal, cosVal, 0], [0, 0, 1]])

    mat2 = np.mat([[leftEye0, leftEye1, 1], [rightEye0, rightEye1, 1], [img_land[2 * 13], img_land[2 * 13 + 1], 1],
                   [img_land[2 * 31], img_land[2 * 31 + 1], 1], [img_land[2 * 37], img_land[2 * 37 + 1], 1]])

    mat2 = (mat1 * mat2.T).T

    cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
    cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5

    if (float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(max(mat2[:, 1]) - min(mat2[:, 1]))):
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 0]) - min(mat2[:, 0])))
    else:
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 1]) - min(mat2[:, 1])))

    scale = (img_size - 1) / 2.0 / halfSize
    mat3 = np.mat([[scale, 0, scale * (halfSize - cx)], [0, scale, scale * (halfSize - cy)], [0, 0, 1]])
    mat = mat3 * mat1

    aligned_img = cv2.warpAffine(img, mat[0:2, :], (img_size, img_size), cv2.INTER_LINEAR, borderValue=(128, 128, 128))

    land_3d = np.ones((int(len(np_shape)/2), 3))
    land_3d[:, 0:2] = np.reshape(np.array(np_shape), (int(len(np_shape)/2), 2))
    mat_land_3d = np.mat(land_3d)
    new_land = np.array((mat * mat_land_3d.T).T)
    new_land = np.reshape(new_land[:, 0:2], len(np_shape))

    return aligned_img, new_land


files = os.listdir(args.image_path)
files.sort()

pbar = tqdm(total=len(files))
for k, f in enumerate(files):
    if f.endswith('.jpg') or f.endswith('.png'):
        fshape = []
        I = cv2.imread(args.image_path + f)
        faceRects = detector(I)
        if len(faceRects) >= 1:
            shape = landmark_predictor(I, faceRects[0])
            for i in range(68):
                fshape.append(shape.part(i).x)
                fshape.append(shape.part(i).y)

            np_shape = np.array(fshape, dtype=np.float)

            aligned_img, new_land = align_face_49pts(I, np_shape, box_enlarge, img_size)

            if not os.path.exists('../dataset/' + crop_name + '/' + f):
                cv2.imwrite('../dataset/' + crop_name + '/' + f, aligned_img)

            name_file.write(f + '\n')
            shape_file.write(str(list(np.around(new_land, 5))) + '\n')

    pbar.update()

name_file.close()
shape_file.close()
pbar.close()
print('done!')
