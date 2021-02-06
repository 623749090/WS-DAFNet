import numpy as np
import random
from PIL import Image
import os
import re
import get_attention_map
import torch
import cv2

patt = re.compile('\d+')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # Image value: [0,1]
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class ImageList(object):
    def __init__(self, crop_size, path, img_path, NUM_CLASS=12, phase='train', transform=None, target_transform=None,
                 loader=default_loader):

        image_list = open(path).readlines()

        self.imgs = []
        for f in image_list:
            fname, flabel, fpos = f.split('->')
            if os.path.exists(img_path + fname):
                self.imgs.append(f)

        self.img_path = img_path
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images in subfolders of: ' + path + '\n'))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.crop_size = crop_size
        self.phase = phase
        self.NUM_CLASS = NUM_CLASS

    def __getitem__(self, index):
        f = self.imgs[index]
        dataps = np.zeros((10, 4))
        fname, flabel, fpos = f.split('->')

        land = fpos[1:-2]
        land = np.array([float(t) for t in land.split(',')])

        img = self.loader(self.img_path + fname)
        if self.phase == 'train':
            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)

            if self.transform is not None:
                img = self.transform(img, flip, offset_x, offset_y)
            if self.target_transform is not None:
                land = self.target_transform(land, flip, offset_x, offset_y)
        # for testing
        else:
            w, h = img.size
            offset_y = (h - self.crop_size)/2
            offset_x = (w - self.crop_size)/2
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                land = self.target_transform(land, 0, offset_x, offset_y)

        feat_map = get_attention_map.get_map_single_au(np.reshape(land, (68, 2)), self.crop_size, self.crop_size)
        feat_map224 = cv2.resize(feat_map, (self.crop_size, self.crop_size))
        img = torch.cat([img, torch.from_numpy(feat_map224).unsqueeze(0)], 0).float()

        dataps[:, :] = get_attention_map.get_au_tg_dlib(np.reshape(land, (68, 2)), self.crop_size, self.crop_size).astype('float32')
        dataps /= 100
        dataps *= 28
        dataps = torch.from_numpy(dataps.astype('int32'))

        au = np.array(patt.findall(flabel)).astype(int)
        au = torch.from_numpy(au.astype('float32'))

        return img, au, dataps

    def __len__(self):
        return len(self.imgs)
