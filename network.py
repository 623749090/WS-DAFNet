from __future__ import print_function
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Enhanced_Model(nn.Module):
    def __init__(self, args):
        super(Enhanced_Model, self).__init__()

        self.vgg19 = models.vgg19(pretrained=False)
        self.vgg19.load_state_dict(torch.load('./data/vgg19-dcbb9e9d.pth'))

        self.conv3_1 = nn.Sequential(*list(self.vgg19.features.children())[: 12])
        self.conv3_4 = nn.Sequential(*list(self.vgg19.features.children())[12: 18])
        self.conv4_1 = nn.Sequential(*list(self.vgg19.features.children())[18: 21])
        self.conv4_4 = nn.Sequential(*list(self.vgg19.features.children())[21: 27])
        self.features = nn.Sequential(*list(self.vgg19.features.children())[27:])

        self._classifier = self.make_classifier(args.NUM_CLASS)
        self.BN = nn.BatchNorm2d(num_features=512, momentum=0.999)

    def preprocess(self, input):
        input_map = input[:, 3:4]

        map112 = F.max_pool2d(input_map, kernel_size=2)
        map56 = F.max_pool2d(map112, kernel_size=2)
        map28 = F.max_pool2d(map56, kernel_size=2)

        map56_256 = torch.cat([map56 for i in range(256)], dim=1)
        map28_512 = torch.cat([map28 for i in range(512)], dim=1)

        x = input[:, 0:3]

        feature3_1 = self.conv3_1(x)
        feature3_4 = self.conv3_4(feature3_1)
        conv3_map = torch.mul(feature3_1, map56_256)
        conv3_all = torch.add(feature3_4, conv3_map)

        feature4_1 = self.conv4_1(conv3_all)
        feature4_4 = self.conv4_4(feature4_1)
        conv4_map = torch.mul(feature4_1, map28_512)
        conv4_all = torch.add(feature4_4, conv4_map)
        return conv4_all

    def make_classifier(self, num_classes):
        return nn.Sequential(
            nn.Sequential(*list(self.vgg19.classifier.children())[:3]),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        conv4_all = self.preprocess(x)
        x = self.features(conv4_all)
        x = x.view(x.size(0), -1)
        net = self._classifier(x)
        return net

    def predict_BN(self, x):
        x = self.preprocess(x)
        x = self.BN(x)
        return x


class Slice_Model(nn.Module):
    def __init__(self, args):
        super(Slice_Model, self).__init__()

        self.roi_upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_roi = nn.Conv2d(in_channels=args.NUM_CLASS, out_channels=args.NUM_CLASS, kernel_size=3)
        self.au_fc = nn.Linear(in_features=args.NUM_CLASS * 4 * 4, out_features=50)
        self.local_fc2 = nn.Linear(in_features=1000, out_features=512)
        self.real_out = nn.Linear(in_features=512, out_features=args.NUM_CLASS)
        self.num_class = args.NUM_CLASS

    def forward(self, input):
        au_fc_layer = []

        for i in range(20):
            x = input[:, i * self.num_class:(i + 1) * self.num_class]
            x = self.roi_upsample(x)
            x = F.relu(self.conv_roi(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.au_fc(x))
            au_fc_layer.append(x)

        net = torch.stack(au_fc_layer)
        net = net.view([-1, 1000])
        net = F.dropout(self.local_fc2(net), p=0.5)
        net = self.real_out(net)
        return net


class Attention_Model(nn.Module):
    def __init__(self, args):
        super(Attention_Model, self).__init__()

        # attention map
        self.att_conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.att_bn1 = nn.BatchNorm2d(num_features=512, momentum=0.999)
        self.att_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.att_bn2 = nn.BatchNorm2d(num_features=512, momentum=0.999)
        self.att_conv3 = nn.Conv2d(in_channels=512, out_channels=args.NUM_CLASS, kernel_size=1)

        # confidence map
        self.se = SELayer(args.NUM_CLASS, reduction=args.REDUCTION)
        self.att_feat_conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.att_feat_bn1 = nn.BatchNorm2d(num_features=512, momentum=0.999)
        self.att_feat_conv2 = nn.Conv2d(in_channels=512, out_channels=args.NUM_CLASS, kernel_size=1)

        self.num_class = args.NUM_CLASS

    def forward(self, input):
        # attention map
        x = self.att_conv1(input)
        x = F.relu(self.att_bn1(x))
        x = self.att_conv2(x)
        x = F.relu(self.att_bn2(x))
        x = self.att_conv3(x)
        x = x.view(x.size(0), self.num_class, -1)
        x = F.softmax(x, dim=2)
        att_map = x.view([-1, self.num_class, 28, 28])

        # confidence map
        y = self.att_feat_conv1(input)
        y = F.relu(self.att_feat_bn1(y))
        conf_map = self.att_feat_conv2(y)

        att_conf = torch.mul(conf_map, att_map)
        att_conf_map = att_conf.sum(3).sum(2)

        att_map_feat = torch.mul(F.sigmoid(conf_map), att_map)
        att_map_feat = self.se(att_map_feat)

        return att_map_feat, att_conf_map


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Fusion_Model(nn.Module):
    def __init__(self, args):
        super(Fusion_Model, self).__init__()
        self.fc = nn.Linear(args.NUM_CLASS, args.NUM_CLASS)

    def forward(self, x):
        x = self.fc(x)
        return x
