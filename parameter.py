"""
@FileName：parameter.py\n
@Description：\n
@Author：WBobby\n
@Department：CUG\n
@Time：2023/6/13 16:10\n
"""
import torch
from coca_pytorch import CoCa
from thop import profile
from torch import nn
from torchsummary import summary
from torchvision.models import vgg16, resnet50
from vit_pytorch.simple_vit_with_patch_dropout import SimpleViT
from vit_pytorch.extractor import Extractor
import timm
from BBnet.model import PVFCNet
from EfficientNET.model import efficientnet_b0
from pytorch.timm.models.mobilevit import mobilevit_s
from timm.models.effnetv2 import effnetv2_s

x = torch.rand(size=(1, 3, 112, 112))
model = timm.create_model('resnet50', pretrained=False, num_classes=10)
# 为网络重写分类层
# model = PVFCNet(11, 5, 10)
# model = effnetv2_s()
# output = model(x)
# print(output.shape)
print(summary(model, (3, 112, 112), device="cpu"))
# print('model_name:coat_small')
# print('flops:{}, params:{}'.format(model, params))
