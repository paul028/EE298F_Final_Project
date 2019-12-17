import torch
import torchvision
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from shutil import copyfile
from PIL import Image
import math

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import cv2

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

import pytorch_ssim
import pytorch_iou
from utils import AverageMeter, VisdomLinePlotter
from paperBASNET.metrics_mod import eval_mae, prec_recall

dataset_name = 'DUTS-TE'
data_dir = os.getenv("HOME") + '/Documents/Courses/EE298-CV/finalproj/datasets/dataset_test/' + dataset_name + '/'
### Place model.pth in "evaluation_models" folder
model_name = 'basnet_orig_pretrained.pth'
### If evaluating structural architecture, set this to false
is_orig_basnet = True


if is_orig_basnet: from paperBASNET.model import BASNet
else: from model import BASNet

def train():
    if os.name == 'nt':
       data_dir =   'C:/Users/marky/Documents/Courses/saliency/datasets/dataset_test/' + dataset_name + '/'
    else:
        data_dir =   os.getenv("HOME") + '/Documents/Courses/EE298-CV/finalproj/datasets/dataset_test/' + dataset_name + '/'
    test_image_dir = 'images/'
    test_label_dir = 'masks/'

    image_ext = '.jpg'
    label_ext = '.png'

    # model_dir = "./saved_models/basnet_bsi_aug/"
    model_dir = "./evaluation_models/"
    resume_train = True ### Just testing
    # resume_model_path = model_dir + "bestMAE/" + model_name
    resume_model_path = model_dir + model_name
    last_epoch = 1
    epoch_num = 100000
    batch_size_train = 8
    batch_size_val = 1
    train_num = 0
    val_num = 0
    enableInpaintAug = False
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    test_img_name_list = glob.glob(data_dir + test_image_dir + '*' + image_ext)
    print("data_dir + test_image_dir + '*' + image_ext: ", data_dir + test_image_dir + '*' + image_ext)

    test_lbl_name_list = []
    for img_path in test_img_name_list:
        img_name = img_path.split("/")[-1]
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]
        test_lbl_name_list.append(data_dir + test_label_dir + imidx + label_ext)

    print("---")
    print("test images: ", len(test_img_name_list))
    print("test labels: ", len(test_lbl_name_list))
    print("---")

    test_num = len(test_img_name_list)
    salobj_dataset_test = SalObjDataset(
        img_name_list=test_img_name_list,
        lbl_name_list=test_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(256),
            RandomCrop(224),
            ToTensorLab(flag=0)]),
            category="test",
            enableInpaintAug=enableInpaintAug)
    salobj_dataloader_test = DataLoader(salobj_dataset_test, batch_size=batch_size_val, shuffle=False, num_workers=1)

    # ------- 3. define model --------
    # define the net
    net = BASNet(3, 1)
    if resume_train:
        # print("resume_model_path:", resume_model_path)
        checkpoint = torch.load(resume_model_path)
        net.load_state_dict(checkpoint)
    if torch.cuda.is_available():
        net.to(device)

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    best_ave_mae = 100000
    best_max_fmeasure = 0
    best_relaxed_fmeasure = 0
    best_ave_maxf = 0
    best_own_RelaxedFmeasure=0
    average_mae = AverageMeter()
    average_maxf = AverageMeter()
    average_relaxedf = AverageMeter()
    average_own_RelaxedFMeasure = AverageMeter()
    average_prec = AverageMeter()
    average_rec = AverageMeter()
    print("---Evaluate model---")
    net.eval()
    max_epoch_fmeasure = 0
    for i, data in enumerate(salobj_dataloader_test):
        inputs, labels = data['image'], data['label']
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        if torch.cuda.is_available(): inputs_v, labels_v = Variable(inputs.to(device), requires_grad=False), Variable(labels.to(device), requires_grad=False)
        else: inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
        if is_orig_basnet: d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs_v)
        else: d0, d1, d2, d3, d4, d5, d6, d7, d1_struct, d2_struct, d3_struct, d4_struct, d5_struct, d6_struct, d7_struct = net(inputs_v)
        maeVal=eval_mae(d0.cpu().data,labels.cpu().data)
        maeVal = eval_mae(d0.cpu().data, labels.cpu().data).item()
        average_mae.update(maeVal, 1)
        prec, recall = prec_recall(d0.cpu().dahttp://localhost:8888/notebooks/evaluation_notebook.ipynb#ta, labels.cpu().data)
        average_prec.update(prec, 1)
        average_rec.update(recall, 1)
        print("[idx: %d, maeVal: %3f, MaxMax: %3f, AveMax: %3f]" % (i, average_mae.avg, average_prec.avg, average_rec.avg))
    beta2 = math.sqrt(0.3)  # for max F_beta metric
    denom = (beta2 ** 2 * average_prec.avg + average_rec.avg)
    if denom == 0: score = 0
    else: score = ((1 + beta2 ** 2) * average_prec.avg * average_rec.avg) / denom
    # score[score != score] = 0  # delete the nan
    print("MAE score: ", average_mae.avg)
    print("MaxF score: ", score)
def run():
    torch.multiprocessing.freeze_support()
    print('loop')

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()
    train()
