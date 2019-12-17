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
from paperBASNET.metrics import getMAE, getPRCurve, getMaxFMeasure, getRelaxedFMeasure, own_RelaxedFMeasure
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

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, d1_struct, d2_struct, d3_struct, d4_struct, d5_struct, d6_struct, d7_struct, labels_v,
								ave_loss0,
								ave_loss1,
								ave_loss2,
								ave_loss3,
								ave_loss4,
								ave_loss5,
								ave_loss6,
								ave_loss7,
								ave_struct_loss1,
								ave_struct_loss2,
								ave_struct_loss3,
								ave_struct_loss4,
								ave_struct_loss5,
								ave_struct_loss6,
								ave_struct_loss7,):

	loss0 = bce_ssim_loss(d0,labels_v)
	loss1 = bce_ssim_loss(d1,labels_v)
	loss2 = bce_ssim_loss(d2,labels_v)
	loss3 = bce_ssim_loss(d3,labels_v)
	loss4 = bce_ssim_loss(d4,labels_v)
	loss5 = bce_ssim_loss(d5,labels_v)
	loss6 = bce_ssim_loss(d6,labels_v)
	loss7 = bce_ssim_loss(d7,labels_v)
	# Struct losses
	loss8 = bce_ssim_loss(d1_struct,labels_v)
	loss9 = bce_ssim_loss(d2_struct,labels_v)
	loss10 = bce_ssim_loss(d3_struct,labels_v)
	loss11 = bce_ssim_loss(d4_struct,labels_v)
	loss12 = bce_ssim_loss(d5_struct,labels_v)
	loss13 = bce_ssim_loss(d6_struct,labels_v)
	loss14 = bce_ssim_loss(d7_struct,labels_v)
	#ssim0 = 1 - ssim_loss(d0,labels_v)

	# iou0 = iou_loss(d0,labels_v)
	#loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + 0.001 * (loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14)#+ 5.0*lossa
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data,loss1.data,loss2.data,loss3.data,loss4.data,loss5.data,loss6.data))
	print("l8: %3f, l9: %3f, l10: %3f, l11: %3f, l2: %3f, l13: %3f, l14: %3f\n"%(loss8.data,loss9.data,loss10.data,loss11.data,loss12.data,loss13.data,loss14.data))
	# print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))

	ave_loss0.update(loss0.data, d0.size(0))
	ave_loss1.update(loss1.data, d1.size(0))
	ave_loss2.update(loss2.data, d2.size(0))
	ave_loss3.update(loss3.data, d3.size(0))
	ave_loss4.update(loss4.data, d4.size(0))
	ave_loss5.update(loss5.data, d5.size(0))
	ave_loss6.update(loss6.data, d6.size(0))
	ave_loss7.update(loss7.data, d7.size(0))
	ave_struct_loss1.update(loss8.data, d1_struct.size(0))
	ave_struct_loss2.update(loss9.data, d2_struct.size(0))
	ave_struct_loss3.update(loss10.data, d3_struct.size(0))
	ave_struct_loss4.update(loss11.data, d4_struct.size(0))
	ave_struct_loss5.update(loss12.data, d5_struct.size(0))
	ave_struct_loss6.update(loss13.data, d6_struct.size(0))
	ave_struct_loss7.update(loss14.data, d7_struct.size(0))

	return loss0, loss



############
############
############
############ GLOBAL PARAMETERS
def train():
    if os.name == 'nt':
       data_dir =   'C:/Users/marky/Documents/Courses/saliency/datasets/DUTS/'
    else:
        data_dir =   os.getenv("HOME") + '/Documents/Courses/EE298-CV/finalproj/datasets/DUTS/'
    test_image_dir = 'DUTS-TE/DUTS-TE-Image/'
    test_label_dir = 'DUTS-TE/DUTS-TE-Mask/'

    image_ext = '.jpg'
    label_ext = '.png'

    model_dir = "./saved_models/basnet_bsi_aug/"
    resume_train = True
    resume_model_path = model_dir + "basnet_bsi_epoch_81_itr_106839_train_1.511335_tar_0.098392.pth"
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

    plotter = VisdomLinePlotter(env_name=visdom_tab_title)

    best_ave_mae = 100000
    best_max_fmeasure = 0
    best_relaxed_fmeasure = 0
    best_ave_maxf = 0
    best_own_RelaxedFmeasure=0
    for epoch in range(last_epoch-1, epoch_num):
    	### Train network
    	train_loss0 = AverageMeter()
    	train_loss1 = AverageMeter()
    	train_loss2 = AverageMeter()
    	train_loss3 = AverageMeter()
    	train_loss4 = AverageMeter()
    	train_loss5 = AverageMeter()
    	train_loss6 = AverageMeter()
    	train_loss7 = AverageMeter()
    	train_struct_loss1 = AverageMeter()
    	train_struct_loss2 = AverageMeter()
    	train_struct_loss3 = AverageMeter()
    	train_struct_loss4 = AverageMeter()
    	train_struct_loss5 = AverageMeter()
    	train_struct_loss6 = AverageMeter()
    	train_struct_loss7 = AverageMeter()

    	test_loss0 = AverageMeter()
    	test_loss1 = AverageMeter()
    	test_loss2 = AverageMeter()
    	test_loss3 = AverageMeter()
    	test_loss4 = AverageMeter()
    	test_loss5 = AverageMeter()
    	test_loss6 = AverageMeter()
    	test_loss7 = AverageMeter()
    	test_struct_loss1 = AverageMeter()
    	test_struct_loss2 = AverageMeter()
    	test_struct_loss3 = AverageMeter()
    	test_struct_loss4 = AverageMeter()
    	test_struct_loss5 = AverageMeter()
    	test_struct_loss6 = AverageMeter()
    	test_struct_loss7 = AverageMeter()

    	average_mae = AverageMeter()
    	average_maxf = AverageMeter()
    	average_relaxedf = AverageMeter()
    	average_own_RelaxedFMeasure = AverageMeter()
    	### Validate model
    	print("---Evaluate model---")
    	if ite_num >= next_test:  # test and save model 10000 iterations, due to very large DUTS-TE dataset
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
    			d0, d1, d2, d3, d4, d5, d6, d7, d1_struct, d2_struct, d3_struct, d4_struct, d5_struct, d6_struct, d7_struct = net(inputs_v)

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
    			result_mae = getMAE(gt_img, resizedImg_np)
    			average_mae.update(result_mae, 1)
    			precision, recall = getPRCurve(gt_img, resizedImg_np)
    			result_maxfmeasure = getMaxFMeasure(precision, recall)
    			result_maxfmeasure = result_maxfmeasure.mean()
    			average_maxf.update(result_maxfmeasure, 1)
    			if (result_maxfmeasure > max_epoch_fmeasure):
    				max_epoch_fmeasure = result_maxfmeasure
    			result_relaxedfmeasure = getRelaxedFMeasure(gt_img, resizedImg_np)
    			result_ownrelaxedfmeasure = own_RelaxedFMeasure(gt_img,resizedImg_np)
    			average_relaxedf.update(result_relaxedfmeasure, 1)
    			average_own_RelaxedFMeasure.update(result_ownrelaxedfmeasure,1)
    			loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, d1_struct, d2_struct, d3_struct, d4_struct, d5_struct, d6_struct, d7_struct, labels_v,
    									test_loss0,
    									test_loss1,
    									test_loss2,
    									test_loss3,
    									test_loss4,
    									test_loss5,
    									test_loss6,
    									test_loss7,
    									test_struct_loss1,
    									test_struct_loss2,
    									test_struct_loss3,
    									test_struct_loss4,
    									test_struct_loss5,
    									test_struct_loss6,
    									test_struct_loss7)
    			del d0, d1, d2, d3, d4, d5, d6, d7, d1_struct, d2_struct, d3_struct, d4_struct, d5_struct, d6_struct, d7_struct, loss2, loss
    			print("[test epoch: %3d/%3d, batch: %5d/%5d, ite: %d] test loss: %3f, tar: %3f " % (epoch + 1, epoch_num, (i + 1) * batch_size_val, test_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
    		model_name = model_dir + "basnet_bsi_epoch_%d_itr_%d_train_%3f_tar_%3f.pth" % (epoch+1, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val)
    		torch.save(net.state_dict(), model_name)
    		running_loss = 0.0
    		running_tar_loss = 0.0
    		net.train()  # resume train
    		ite_num4val = 1
    		if (average_mae.avg < best_ave_mae):
    			best_ave_mae = average_mae.avg
    			newname = model_dir + "bestMAE/basnet_bsi_epoch_%d_itr_%d_train_%3f_tar_%3f_mae_%3f.pth" % (epoch+1, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val, best_ave_mae)
    			fold_dir = newname.rsplit("/", 1)
    			if not os.path.isdir(fold_dir[0]): os.mkdir(fold_dir[0])
    			copyfile(model_name, newname)
    		if (max_epoch_fmeasure > best_max_fmeasure):
    			best_max_fmeasure = max_epoch_fmeasure
    			newname = model_dir + "bestEpochMaxF/basnet_bsi_epoch_%d_itr_%d_train_%3f_tar_%3f_maxfmeas_%3f.pth" % (epoch+1, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val, best_max_fmeasure)
    			fold_dir = newname.rsplit("/", 1)
    			if not os.path.isdir(fold_dir[0]): os.mkdir(fold_dir[0])
    			copyfile(model_name, newname)
    		if (average_maxf.avg > best_ave_maxf):
    			best_ave_maxf = average_maxf.avg
    			newname = model_dir + "bestAveMaxF/basnet_bsi_epoch_%d_itr_%d_train_%3f_tar_%3f_avemfmeas_%3f.pth" % (epoch+1, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val, best_ave_maxf)
    			fold_dir = newname.rsplit("/", 1)
    			if not os.path.isdir(fold_dir[0]): os.mkdir(fold_dir[0])
    			copyfile(model_name, newname)
    		# if (average_relaxedf.avg > best_relaxed_fmeasure):
    		# 	best_relaxed_fmeasure = average_relaxedf.avg
    		# 	newname = model_dir + "bestAveRelaxF/basnet_bsi_epoch_%d_itr_%d_train_%3f_tar_%3f_averelaxfmeas_%3f.pth" % (epoch+1, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val, best_relaxed_fmeasure)
    		# 	fold_dir = newname.rsplit("/", 1)
    		# 	if not os.path.isdir(fold_dir[0]): os.mkdir(fold_dir[0])
    		# 	copyfile(model_name, newname)
    		# if(average_own_RelaxedFMeasure.avg > best_own_RelaxedFmeasure):
    		#     best_own_RelaxedFmeasure=average_own_RelaxedFMeasure.avg
    		#     newname = model_dir + "bestOwnRelaxedF/basnet_bsi_epoch_%d_itr_%d_train_%3f_tar_%3f_averelaxfmeas_%3f.pth" % (epoch+1, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val, best_own_RelaxedFmeasure)
    		#     fold_dir = newname.rsplit("/", 1)
    		#     if not os.path.isdir(fold_dir[0]): os.mkdir(fold_dir[0])
    		#     copyfile(model_name, newname)
    		plotter.plot('loss0', 'test', 'Main Loss 0', epoch+1, float(test_loss0.avg))
    		plotter.plot('loss1', 'test', 'Main Loss 1', epoch+1, float(test_loss1.avg))
    		plotter.plot('loss2', 'test', 'Main Loss 2', epoch+1, float(test_loss2.avg))
    		plotter.plot('loss3', 'test', 'Main Loss 3', epoch+1, float(test_loss3.avg))
    		plotter.plot('loss4', 'test', 'Main Loss 4', epoch+1, float(test_loss4.avg))
    		plotter.plot('loss5', 'test', 'Main Loss 5', epoch+1, float(test_loss5.avg))
    		plotter.plot('loss6', 'test', 'Main Loss 6', epoch+1, float(test_loss6.avg))
    		plotter.plot('loss7', 'test', 'Main Loss 7', epoch+1, float(test_loss7.avg))
    		plotter.plot('structloss1', 'test', 'Struct Loss 1', epoch+1, float(test_struct_loss1.avg))
    		plotter.plot('structloss2', 'test', 'Struct Loss 2', epoch+1, float(test_struct_loss2.avg))
    		plotter.plot('structloss3', 'test', 'Struct Loss 3', epoch+1, float(test_struct_loss3.avg))
    		plotter.plot('structloss4', 'test', 'Struct Loss 4', epoch+1, float(test_struct_loss4.avg))
    		plotter.plot('structloss5', 'test', 'Struct Loss 5', epoch+1, float(test_struct_loss5.avg))
    		plotter.plot('structloss6', 'test', 'Struct Loss 6', epoch+1, float(test_struct_loss6.avg))
    		plotter.plot('structloss7', 'test', 'Struct Loss 7', epoch+1, float(test_struct_loss7.avg))
    		plotter.plot('mae', 'test', 'Average Epoch MAE', epoch+1, float(average_mae.avg))
    		plotter.plot('max_maxf', 'test', 'Max Max Epoch F-Measure', epoch+1, float(max_epoch_fmeasure))
    		plotter.plot('ave_maxf', 'test', 'Average Max F-Measure', epoch+1, float(average_maxf.avg))
    		# plotter.plot('ave_relaxedf', 'test', 'Average Relaxed F-Measure', epoch+1, float(average_relaxedf.avg))
    		# plotter.plot('own_RelaxedFMeasure','test','Own Average Relaxed F-Measure', epoch+1, float(average_own_RelaxedFMeasure.avg))
    print('-------------Congratulations! Training Done!!!-------------')

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()
    train()
