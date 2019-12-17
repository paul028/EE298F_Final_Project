import torch
import torchvision
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from shutil import copyfile
from PIL import Image

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

from model import BASNet

import pytorch_ssim
import pytorch_iou
from utils import AverageMeter, VisdomLinePlotter
from metrics import getMAE, getPRCurve, getMaxFMeasure, getRelaxedFMeasure

# ------- 1. define loss function --------
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

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target):

	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)
	iou_out = iou_loss(pred,target)

	loss = bce_out + ssim_out + iou_out

	return loss

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v,ave_loss0,
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


# ------- 2. set the directory of training dataset --------
def train():
    data_dir = '/media/markytools/New Volume/Courses/EE298CompVis/finalproject/datasets/'
    test_image_dir = 'DUTS/DUTS-TE/DUTS-TE-Image/'
    test_label_dir = 'DUTS/DUTS-TE/DUTS-TE-Mask/'
    image_ext = '.jpg'
    label_ext = '.png'

    model_dir = "../saved_models/"
    resume_train = True
    resume_model_path = model_dir + "basnet-original.pth"
    last_epoch = 1
    epoch_num = 100000
    batch_size_train = 8
    batch_size_val = 1
    train_num = 0
    val_num = 0
    enableInpaintAug = False
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") #set CPU to 0
    # ------- 5. training process --------
    print("---start training...")
    test_increments = 15000
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 1
    next_test = ite_num + 0
    ############
    ############
    ############
    ############
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
    for test_lbl in test_lbl_name_list:
        test_jpg = test_lbl.replace("png", "jpg")
        test_jpg = test_jpg.replace("Mask", "Image")
        if test_jpg not in test_img_name_list: print("test_lbl not in label: ", test_lbl)

    salobj_dataset_test = SalObjDataset(
        img_name_list=test_img_name_list,
        lbl_name_list=test_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(256),
            RandomCrop(224),
            ToTensorLab(flag=0)]),
    		category="test",
    		enableInpaintAug=enableInpaintAug)
    salobj_dataloader_test = DataLoader(salobj_dataset_test, batch_size=batch_size_val, shuffle=True, num_workers=1)

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

    plotter = VisdomLinePlotter(env_name='NewlyAddedRelaxedMeasureEnv1')

    best_ave_mae = 100000
    best_max_fmeasure = 0
    best_relaxed_fmeasure = 0
    best_ave_maxf = 0

    ### Train network
    train_loss0 = AverageMeter()
    train_loss1 = AverageMeter()
    train_loss2 = AverageMeter()
    train_loss3 = AverageMeter()
    train_loss4 = AverageMeter()
    train_loss5 = AverageMeter()
    train_loss6 = AverageMeter()
    train_loss7 = AverageMeter()


    test_loss0 = AverageMeter()
    test_loss1 = AverageMeter()
    test_loss2 = AverageMeter()
    test_loss3 = AverageMeter()
    test_loss4 = AverageMeter()
    test_loss5 = AverageMeter()
    test_loss6 = AverageMeter()
    test_loss7 = AverageMeter()

    average_mae = AverageMeter()
    average_maxf = AverageMeter()
    average_relaxedf = AverageMeter()
    ### Validate model
    print("---Evaluate model---")
    next_test = ite_num + test_increments
    net.eval()
    max_epoch_fmeasure = 0
    for i, data in enumerate(salobj_dataloader_test):
        inputs, labels = data['image'], data['label']
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.to(device), requires_grad=False), Variable(labels.to(device), requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
        d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs_v)
        pred = d0[:,0,:,:]
        pred = normPRED(pred)
        pred = pred.squeeze()
        predict_np = pred.cpu().data.numpy()
        im = Image.fromarray(predict_np*255).convert('RGB')
        img_name = test_img_name_list[i]
        image = cv2.imread(img_name)
        imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
        imo = imo.convert("L") ###  Convert to grayscale 1-channel
        resizedImg_np = np.array(imo) ### Result is 2D numpy array predicted salient map
        img__lbl_name = test_lbl_name_list[i]
        gt_img = np.array(Image.open(img__lbl_name).convert("L")) ### Ground truth salient map

        ### Compute metrics
        img_name_png =
        result_mae = getMAE(gt_img, resizedImg_np)
        average_mae.update(result_mae, 1)
        precision, recall = getPRCurve(gt_img, resizedImg_np)
        result_maxfmeasure = getMaxFMeasure(precision, recall)
        result_maxfmeasure = result_maxfmeasure.mean()
        average_maxf.update(result_maxfmeasure, 1)
        if (result_maxfmeasure > max_epoch_fmeasure):
        	max_epoch_fmeasure = result_maxfmeasure
        result_relaxedfmeasure = getRelaxedFMeasure(gt_img, resizedImg_np)
        average_relaxedf.update(result_relaxedfmeasure, 1)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7,labels_v,
        						test_loss0,
        						test_loss1,
        						test_loss2,
        						test_loss3,
        						test_loss4,
        						test_loss5,
        						test_loss6,
        						test_loss7)
        del d0, d1, d2, d3, d4, d5, d6, d7,loss2, loss
    print("Average Epoch MAE: ", average_mae.avg)
    print("Max Max Epoch F-Measure: ", average_maxf.avg)
    print("Average Max F-Measure: ", max_epoch_fmeasure)
    print("Average Relaxed F-Measure: ", average_relaxedf.avg)

    print('-------------Congratulations! Training Done!!!-------------')

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()
    train()
