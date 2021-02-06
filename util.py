from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import init


def prep_model_input(feat_data, pos_para):
    SIZE = 3
    NUM_FEAT = 12
    new_input = torch.autograd.Variable(torch.zeros(feat_data.size(0), 20 * NUM_FEAT, SIZE, SIZE)).cuda()

    for i in range(feat_data.size(0)):
        for j in range(10):
            p = pos_para[i, j, :].data

            for k in range(4):
                p[k] = SIZE//2 if p[k] < SIZE//2 else p[k]
                p[k] = 27 - SIZE//2 if p[k] > 27 - SIZE//2 else p[k]
            try:
                new_input[i, NUM_FEAT * 2 * j:NUM_FEAT * (2 * j + 1), :, :] = feat_data[i, :, p[1] - SIZE//2:p[1] + SIZE//2+1,
                                                                                p[0] - SIZE//2:p[0] + SIZE//2+1]
                new_input[i, NUM_FEAT * (2 * j + 1):NUM_FEAT * (2 * j + 2), :, :] = feat_data[i, :, p[3] - SIZE//2:p[3] + SIZE//2+1,
                                                                                      p[2] - SIZE//2:p[2] + SIZE//2+1]
            except Exception as e:
                print(p)

    return new_input


def get_f1(outputs, y_labels):
    outputs = F.sigmoid(outputs)
    outputs_i = outputs + 0.5
    outputs_i = outputs_i.type(torch.int32)
    y_ilab = y_labels.type(torch.int32)
    # used in f1
    gd_num = torch.sum(y_ilab, dim=0)
    pr_num = torch.sum(outputs_i, dim=0)

    sum_ones = y_ilab + outputs_i
    pr_rtm = sum_ones // 2

    pr_rt = torch.sum(pr_rtm, dim=0)

    # prevent nan to destroy the f1
    pr_rt = pr_rt.type(torch.float32)
    gd_num = gd_num.type(torch.float32)
    pr_num = pr_num.type(torch.float32)

    zero_scale = torch.zeros_like(torch.min(pr_rt))

    if torch.eq(zero_scale, torch.min(gd_num)):
        gd_num += 1
    if torch.eq(zero_scale, torch.min(pr_num)):
        pr_num += 1
    if torch.eq(zero_scale, torch.min(pr_rt)):
        pr_rt += 0.01

    recall = pr_rt / gd_num
    precision = pr_rt / pr_num
    f1 = 2 * recall * precision / (recall + precision)

    return f1


def get_acc(outputs, y_labels):
    outputs = F.sigmoid(outputs)
    outputs_i = outputs + 0.5
    outputs_i = outputs_i.type(torch.int32)
    y_ilab = y_labels.type(torch.int32)
    # used in acc
    pr_rtm = torch.eq(outputs_i, y_ilab)
    pr_rt = torch.sum(pr_rtm, dim=0)
    pr_rt = pr_rt.type(torch.float32)

    acc = pr_rt / outputs.shape[0]

    return acc


def init_netParams(net):
    for name, m in net.named_modules():
        # print(name, type(m))
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def load_param(model, weight_path):
    weights = torch.load(weight_path, map_location=lambda storage, loc: storage)

    for i in range(len(weights)):
        weight_dict = weights[i].state_dict()
        model_dict = model[i].state_dict()
        weight_dict2 = {k: v for k, v in weight_dict.items() if k in model_dict}

        model_dict.update(weight_dict2)
        model[i].load_state_dict(model_dict)


def lr_change(epoch, optimizer):
    if epoch % 2 == 0:
        optimizer.param_groups[0]["lr"] *= 0.5
