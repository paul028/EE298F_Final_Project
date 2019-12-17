import os
import torch
from torchvision import models
from resnet_model import *
# extract vgg features
if __name__ == '__main__':
    save_fold = '../weights'
    if not os.path.exists(save_fold):
        os.mkdir(save_fold)
    resnet = models.resnet50(pretrained=True)
    torch.save(resnet.state_dict(), os.path.join(save_fold, 'resnet101.pth'))
