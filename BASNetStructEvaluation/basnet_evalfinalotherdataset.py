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

from paperBASNET.data_loader import Rescale
from paperBASNET.data_loader import RescaleT
from paperBASNET.data_loader import RandomCrop
from paperBASNET.data_loader import CenterCrop
from paperBASNET.data_loader import ToTensor
from paperBASNET.data_loader import ToTensorLab
from paperBASNET.data_loader import SalObjDataset

from paperBASNET.model import BASNet

import pytorch_ssim
import pytorch_iou
from utils import AverageMeter, VisdomLinePlotter
from paperBASNET.metrics_mod import eval_mae, prec_recall
# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def bce_ssim_loss(pred,target):

    bce_out = bce_loss(pred,target)
    ssim_out = 1 - ssim_loss(pred,target)
    iou_out = iou_loss(pred,target)

    loss = bce_out + ssim_out + iou_out

    return loss

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v,
                                ave_loss0,
                                ave_loss1,
                                ave_loss2,
                                ave_loss3,
                                ave_loss4,
                                ave_loss5,
                                ave_loss6,
                                ave_loss7):

    loss0 = bce_ssim_loss(d0,labels_v)
    loss1 = bce_ssim_loss(d1,labels_v)
    loss2 = bce_ssim_loss(d2,labels_v)
    loss3 = bce_ssim_loss(d3,labels_v)
    loss4 = bce_ssim_loss(d4,labels_v)
    loss5 = bce_ssim_loss(d5,labels_v)
    loss6 = bce_ssim_loss(d6,labels_v)
    loss7 = bce_ssim_loss(d7,labels_v)
    #ssim0 = 1 - ssim_loss(d0,labels_v)

    # iou0 = iou_loss(d0,labels_v)
    #loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7#+ 5.0*lossa
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data,loss1.data,loss2.data,loss3.data,loss4.data,loss5.data,loss6.data))
    # print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))
    ave_loss0.update(loss0.data, d0.size(0))
    ave_loss1.update(loss1.data, d1.size(0))
    ave_loss2.update(loss2.data, d2.size(0))
    ave_loss3.update(loss3.data, d3.size(0))
    ave_loss4.update(loss4.data, d4.size(0))
    ave_loss5.update(loss5.data, d5.size(0))
    ave_loss6.update(loss6.data, d6.size(0))
    ave_loss7.update(loss7.data, d7.size(0))

    return loss0, loss



############
############
############
############ GLOBAL PARAMETERS
def train():
    if os.name == 'nt':
       data_dir =   'C:/Users/marky/Documents/Courses/saliency/datasets/dataset_test/ECSSD/'
    else:
        data_dir =   os.getenv("HOME") + '/Documents/Courses/EE298-CV/finalproj/datasets/dataset_test/ECSSD/'
    test_image_dir = 'image/'
    test_label_dir = 'gt/'

    image_ext = '.jpg'
    label_ext = '.png'

    model_dir = "./saved_models/basnet_bsi_orig/"
    resume_train = True ### Just testing
    resume_model_path = model_dir + "basnet.pth"
    last_epoch = 1
    epoch_num = 100000
    batch_size_train = 8
    batch_size_val = 1
    train_num = 0
    val_num = 0
    enableInpaintAug = False
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # ------- 5. training process --------
    print("---start training...")
    test_increments = 6250
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 1
    next_test = ite_num + 0
    visdom_tab_title = "StructArchWithoutStructImgs(WithHFlip)"

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
        d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs_v)
        maeVal=eval_mae(d0.cpu().data,labels.cpu().data)
        maeVal = eval_mae(d0.cpu().data, labels.cpu().data).item()
        average_mae.update(maeVal, 1)
        prec, recall = prec_recall(d0.cpu().data, labels.cpu().data)
        average_prec.update(prec, 1)
        average_rec.update(recall, 1)
        print("[idx: %d, maeVal: %3f, MaxMax: %3f, AveMax: %3f]" % (i, average_mae.avg, average_prec.avg, average_rec.avg))
    beta2 = math.sqrt(0.3)  # for max F_beta metric
    score = (1 + beta2 ** 2) * average_prec.avg * average_rec.avg / (beta2 ** 2 * average_prec.avg + average_rec.avg)
    score[score != score] = 0  # delete the nan
    print("MAE score: ", average_mae.avg)
    print("MaxF score: ", score)

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()
    train()
