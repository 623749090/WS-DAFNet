from __future__ import print_function
import numpy as np
from tqdm import tqdm

name = '../source/BP4D_crop_name.txt'
shape = '../source/BP4D_crop_shape.txt'

names = open(name).readlines()
shapes = open(shape).readlines()

for operation in ['tr', 'ts']:
    for fold in range(1, 4):

        path = '../source/original_data/BP4D_' + operation + str(fold) + '_path.txt'
        au_path = '../source/original_data/BP4D_' + operation + str(fold) + '_au.txt'
        new_source = open('../source/BP4D_crop_new_' + operation + str(fold) + '.txt', 'w')
        print('Creat BP4D_crop_new_' + operation + str(fold) + '.txt ...')

        au = np.loadtxt(au_path)
        imgs = open(path).readlines()

        pbar = tqdm(total=len(imgs))
        for k, f in enumerate(imgs):
            if f in names:
                fshape = shapes[names.index(f)]
                flabel = str(list(au[k].astype(int)))
                new_source.write(f.strip("\n") + '->' + flabel + '->' + fshape)
            pbar.update()

        new_source.close()
        pbar.close()

print('done')
