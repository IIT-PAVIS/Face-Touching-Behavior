#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
	IIT : Istituto italiano di tecnologia

    Pattern Analysis and Computer Vision (PAVIS) research line

    Usage Example:
		$ python3

    LICENSE:
	This project is licensed under the terms of the MIT license.
	This project incorporates material from the projects listed below (collectively, "Third Party Code").
	This Third Party Code is licensed to you under their original license terms.
	We reserves all other rights not expressly granted, whether by implication, estoppel or otherwise.
	The software can be freely used for any non-commercial applications.
"""

import torch
from torch import nn
from torchvision import models

# Defining the Model
class Resnet152_fc(nn.Module):

    def __init__(self, num_classes):
        super(Resnet152_fc, self).__init__()
        net = models.resnet152(pretrained=True)

        self.features = nn.Sequential(*list(net.children())[:-1])
        self.classifier = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1))
        self.modelName = 'resnet'

    def forward(self, images):
        # x = self.net(images)
        x = self.features(images)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

class SqueezeNet_fc(nn.Module):

    def __init__(self, num_classes):
        super(SqueezeNet_fc, self).__init__()
        net = torch.hub.load('pytorch/vision:v0.5.0', 'squeezenet1_0', pretrained=True)

        self.features = nn.Sequential(*list(net.children())[:-1])
        self.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                        nn.Conv2d(512, num_classes, kernel_size=1),
                                        nn.ReLU(inplace=True),
                                        nn.AvgPool2d(12))

        self.modelName = 'squeezenet'

    def forward(self, images):
        x = self.features(images)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x