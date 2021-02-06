from __future__ import print_function
import numpy as np
import random
import argparse
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from data_list import ImageList
from network import Enhanced_Model, Slice_Model, Attention_Model, Fusion_Model
import pre_process as prep
from util import *

def parse_args():
    parser = argparse.ArgumentParser(description='WS-DAFNet')
    # general
    parser.add_argument('--IM_SIZE', default='224', type=int, help='crop size for images')
    parser.add_argument('--NUM_CLASS', default='12', type=int, help='AU number')
    parser.add_argument('--Train_BATCH', default='36', type=int, help='train batch size')
    parser.add_argument('--Test_BATCH', default='24', type=int, help='test batch size')
    parser.add_argument('--SEED', default='4', type=int, help='random seed number')
    parser.add_argument('--gpus', default='0', type=str, help='GPU device to train with')
    parser.add_argument('--REDUCTION', default='6', type=int, help='reduction in channel attention')

    parser.add_argument('--img_path', default='./dataset/BP4D_crop/', type=str, help='cropped image path')
    parser.add_argument('--listtrainpath', default='source/BP4D_crop_new_tr1.txt', type=str, help='training data path')
    parser.add_argument('--listtestpath', default='source/BP4D_crop_new_ts1.txt', type=str, help='test data path')
    parser.add_argument('--weight_path', default='source/BP4D_crop_new_tr1_weight.txt', type=str, help='weight path for cross entropy loss')
    parser.add_argument('--version', default='Train_DAFNet_fold1', type=str, help='version name')
    parser.add_argument('--name', default='eve2_seed4_tr36', type=str, help='task name')

    parser.add_argument('--lr', default="1e-4", type=float, help='initial learning rate')
    parser.add_argument('--decay', default='5e-4', type=float, help='weight decay for Adam optimizer')
    parser.add_argument('--epochs', default='2', type=int, help='nums of epochs')
    args = parser.parse_args()
    return args
args = parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

if args.SEED > 0:
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)

if not os.path.exists('./result/res/' + args.version):
    os.makedirs('./result/res/' + args.version)
if not os.path.exists('./result/snap/' + args.version):
    os.makedirs('./result/snap/' + args.version)

e_net = Enhanced_Model(args)
e_net = e_net.cuda()

s_net = Slice_Model(args)
s_net = s_net.cuda()
init_netParams(s_net)

a_net = Attention_Model(args)
a_net = a_net.cuda()
init_netParams(a_net)

fusion = Fusion_Model(args)
fusion = fusion.cuda()
init_netParams(fusion)

optimizer = Adam(list(e_net.parameters()) + list(s_net.parameters()) + list(a_net.parameters()) + list(fusion.parameters()),
                 lr=args.lr, weight_decay=args.decay)

au_weight_src = torch.from_numpy(np.loadtxt(args.weight_path)).float().cuda()

# 修改存储结果文件路径
fout_test_f1 = open('./result/res/' + args.version + '/WS-DAFNet_' + args.name + '_f1.txt', 'w')
fout_test_f1_mean = open('./result/res/' + args.version + '/WS-DAFNet_' + args.name + '_f1_mean.txt', 'w')
fout_test_acc = open('./result/res/' + args.version + '/WS-DAFNet_' + args.name + '_acc.txt', 'w')
fout_test_acc_mean = open('./result/res/' + args.version + '/WS-DAFNet_' + args.name + '_acc_mean.txt', 'w')
fout_test = open('./result/res/' + args.version + '/WS-DAFNet_' + args.name + '_predata.txt', 'w')

def perform_operation(file_path, operation, epoch):
    if operation == 'Train':
        torch.enable_grad()
        e_net.train()
        a_net.train()
        s_net.train()
        fusion.train()
    else:
        torch.no_grad()
        e_net.eval()
        a_net.eval()
        s_net.eval()
        fusion.eval()

    if operation == 'Train':
        imDataset = ImageList(crop_size=args.IM_SIZE, path=file_path, img_path=args.img_path, NUM_CLASS=args.NUM_CLASS,
                  phase='test', transform=prep.image_test(crop_size=args.IM_SIZE),
                  target_transform=prep.land_transform(img_size=args.IM_SIZE))
        imDataLoader = torch.utils.data.DataLoader(imDataset, batch_size=args.Train_BATCH, shuffle=True, num_workers=0)
    else:
        imDataset = ImageList(crop_size=args.IM_SIZE, path=file_path, img_path=args.img_path, NUM_CLASS=args.NUM_CLASS,
                  phase='test', transform=prep.image_test(crop_size=args.IM_SIZE),
                  target_transform=prep.land_transform(img_size=args.IM_SIZE))
        imDataLoader = torch.utils.data.DataLoader(imDataset, batch_size=args.Test_BATCH, num_workers=0)

    for batch_Idx, data in enumerate(imDataLoader):
        if operation == 'Train':
            print('%s Epoch: %d Batch_Idx: %d' % (operation, epoch, batch_Idx))

        if operation == 'Train':
            optimizer.zero_grad()

        datablob, datalb, pos_para = data

        datablob = torch.autograd.Variable(datablob).cuda()
        y_lb = torch.autograd.Variable(datalb).view(datalb.size(0), -1).cuda()
        pos_para = torch.autograd.Variable(pos_para).cuda()

        bceLoss_cls = nn.BCEWithLogitsLoss()
        bceLoss2_att = nn.BCEWithLogitsLoss()

        pred_global = e_net(datablob)
        feat_data = e_net.predict_BN(datablob)
        pred_att_map, pred_conf = a_net(feat_data)
        slice_feat_data = prep_model_input(pred_att_map, pos_para)
        pred_local = s_net(slice_feat_data)
        cls_pred = fusion(pred_global + pred_local)

        cls_loss = bceLoss_cls(cls_pred, y_lb)
        att_loss = bceLoss2_att(pred_conf, y_lb)
        sum_loss = cls_loss + att_loss

        if operation == 'Train':
            sum_loss.backward()

        cls_pred = cls_pred.data.cpu().float()
        y_lb = y_lb.data.cpu().float()
        f1_score = get_f1(cls_pred, y_lb)
        acc_scr = get_acc(cls_pred, y_lb)

        if operation == 'Test':
            if batch_Idx == 0:
                all_output = cls_pred
                all_label = y_lb
            else:
                all_output = torch.cat((all_output, cls_pred), 0)
                all_label = torch.cat((all_label, y_lb), 0)

        if operation == 'Train':
            print('acc_scr', acc_scr.mean().cpu().data.item(), 'f1_score', f1_score.mean().cpu().data.item(), 'sum_loss', sum_loss.cpu().data.item())

        if operation == 'Train':
            optimizer.step()

        if operation == 'Test':
            fout_test.write('Label:' + str(y_lb) + '->' + 'Pre:' + str(cls_pred) + '\n')

        del datablob, y_lb, pos_para, feat_data, pred_att_map, pred_conf, slice_feat_data, pred_local, cls_pred, cls_loss, att_loss, sum_loss, acc_scr, f1_score

    if operation == 'Test':
        all_acc_scr = get_acc(all_output, all_label)
        all_f1_score = get_f1(all_output, all_label)

        fout_test_f1.write('***' + str(all_f1_score.numpy().tolist()) + '\n')
        fout_test_f1_mean.write('***' + str(all_f1_score.mean().numpy().tolist()) + '\n')
        fout_test_acc.write('***' + str(all_acc_scr.numpy().tolist()) + '\n')
        fout_test_acc_mean.write('***' + str(all_acc_scr.mean().numpy().tolist()) + '\n')

        print('average f1 score: ', str(all_f1_score.mean().numpy().tolist()))
        print('average acc score: ', str(all_acc_scr.mean().numpy().tolist()))

        del all_acc_scr, all_f1_score, all_output, all_label

    if operation == 'Train':
        new_model = './result/snap/' + args.version + '/WS-DAFNet_' + args.name + '_' + str(epoch) + '.pth'
        torch.save([e_net, a_net, s_net, fusion], new_model)
        print('save ' + new_model)

def main():
    for epoch in range(1, args.epochs+1):

        print('Epoch: %d' % epoch)

        print('Training')
        perform_operation(args.listtrainpath, 'Train', epoch)

        print('Testing...')
        perform_operation(args.listtestpath, 'Test', epoch)

if __name__ == '__main__':
    main()
    print('Done!')
    fout_test_f1.close()
    fout_test_f1_mean.close()
    fout_test_acc.close()
    fout_test_acc_mean.close()
    fout_test.close()
