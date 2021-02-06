from __future__ import print_function
import numpy as np
import random
import argparse
import os
import sys
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from data_list import ImageList
from network import Enhanced_Model, Slice_Model, Attention_Model, Fusion_Model
import pre_process as prep
from tqdm import tqdm
from util import *
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='WS-DAFNet')
    # general
    parser.add_argument('--IM_SIZE', default='224', type=int, help='crop size for images')
    parser.add_argument('--NUM_CLASS', default='12', type=int, help='AU number')
    parser.add_argument('--Test_BATCH', default='30', type=int, help='test batch size')
    parser.add_argument('--gpus', default='0', type=str, help='GPU device to train with')
    parser.add_argument('--resume', default='0', type=int, help='checkpoint to load')
    parser.add_argument('--REDUCTION', default='6', type=int, help='reduction in channel attention')

    parser.add_argument('--resume_path', default='./data/model/WS-DAFNet_fold1.pth', type=str, help='load model path')
    parser.add_argument('--img_path', default='./dataset/BP4D_crop/', type=str, help='cropped image path')
    parser.add_argument('--listtestpath', default='source/BP4D_crop_new_ts1.txt', type=str, help='test data path')
    parser.add_argument('--version', default='Train_DAFNet_fold1', type=str, help='resume version name')
    parser.add_argument('--name', default='eve2_seed4_tr36', type=str, help='resume task name')
    args = parser.parse_args()
    return args
args = parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

e_net = Enhanced_Model(args)
e_net = e_net.cuda()

s_net = Slice_Model(args)
s_net = s_net.cuda()

a_net = Attention_Model(args)
a_net = a_net.cuda()

fusion = Fusion_Model(args)
fusion = fusion.cuda()

model = [e_net, a_net, s_net, fusion]
if args.resume:
    load_param(model, './result/snap/' + args.version + '/WS-DAFNet_' + args.name + '_' + str(args.resume) + '.pth')
    print('resume: ' + './result/snap/' + args.version + '/WS-DAFNet_' + args.name + '_' + str(args.resume) + '.pth')
elif args.resume_path:
    load_param(model, args.resume_path)
    print('resume: ' + str(args.resume_path))
else:
    print('Please select the pre-trained model! !')
    sys.exit()

def perform_operation(file_path):
    torch.no_grad()
    e_net.eval()
    a_net.eval()
    s_net.eval()
    fusion.eval()

    imDataset = ImageList(crop_size=args.IM_SIZE, path=file_path, img_path=args.img_path, NUM_CLASS=args.NUM_CLASS,
              phase='test', transform=prep.image_test(crop_size=args.IM_SIZE),
              target_transform=prep.land_transform(img_size=args.IM_SIZE))
    imDataLoader = torch.utils.data.DataLoader(imDataset, batch_size=args.Test_BATCH, num_workers=0)

    pbar = tqdm(total=len(imDataLoader))
    for batch_Idx, data in enumerate(imDataLoader):

        datablob, datalb, pos_para = data
        datablob = torch.autograd.Variable(datablob).cuda()
        y_lb = torch.autograd.Variable(datalb).view(datalb.size(0), -1).cuda()
        pos_para = torch.autograd.Variable(pos_para).cuda()

        pred_global = e_net(datablob)
        feat_data = e_net.predict_BN(datablob)
        pred_att_map, pred_conf = a_net(feat_data)
        slice_feat_data = prep_model_input(pred_att_map, pos_para)
        pred_local = s_net(slice_feat_data)
        cls_pred = fusion(pred_global + pred_local)

        cls_pred = cls_pred.data.cpu().float()
        y_lb = y_lb.data.cpu().float()

        if batch_Idx == 0:
            all_output = cls_pred
            all_label = y_lb
        else:
            all_output = torch.cat((all_output, cls_pred), 0)
            all_label = torch.cat((all_label, y_lb), 0)
        pbar.update()

    pbar.close()
    all_acc_scr = get_acc(all_output, all_label)
    all_f1_score = get_f1(all_output, all_label)

    print('f1 score: ', str(all_f1_score.numpy().tolist()))
    print('average f1 score: ', str(all_f1_score.mean().numpy().tolist()))
    print('acc score: ', str(all_acc_scr.numpy().tolist()))
    print('average acc score: ', str(all_acc_scr.mean().numpy().tolist()))


def main():

    print('Testing...')
    perform_operation(args.listtestpath)


if __name__ == '__main__':
    main()
    print('Done!')
