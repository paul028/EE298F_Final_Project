import argparse
import os
#from dataset import get_loader
from solver import Solver
import glob
from torchvision import transforms, utils
import torchvision.transforms as standard_transforms

from torch.utils.data import Dataset, DataLoader
from own_dataloader import Rescale
from own_dataloader import RescaleT
from own_dataloader import RandomCrop
from own_dataloader import CenterCrop
from own_dataloader import ToTensor
from own_dataloader import ToTensorLab
from own_dataloader import SalObjDataset

#data_dir =   'C:/Users/tip/Documents/GitHub/Saliency_Dataset/dataset_test/DUTOMRON/'
#data_dir =   'C:/Users/tip/Documents/GitHub/Saliency_Dataset/dataset_test/DUTS-TE/'
data_dir =   'C:/Users/tip/Documents/GitHub/Saliency_Dataset/dataset_test/ECSSD/'
#data_dir =   'C:/Users/tip/Documents/GitHub/Saliency_Dataset/dataset_test/HKU-IS/'
#data_dir =   'C:/Users/tip/Documents/GitHub/Saliency_Dataset/dataset_test/PASCALS/'
#data_dir =   'C:/Users/tip/Documents/GitHub/Saliency_Dataset/dataset_test/SOD/'
#tra_image_dir = 'DUTS-TR/DUTS-TR-Image/'
#tra_label_dir = 'DUTS-TR/DUTS-TR-Mask/'
test_image_dir = 'Imgs/'
test_label_dir = 'gt/'
enableInpaintAug = False
batch_size_train=32
batch_size_val=4

image_ext = '.jpg'
label_ext = '.png'

vgg_path = 'D:/nonat project/Experiment 1/weights/vgg16_feat.pth'
trained_model='D:/nonat project/Experiment 1/weights/Experiment1.pth'

#test_folder='D:/nonat project/Experiment 1/weights/test/DUTOMRON TEST'
#test_folder='D:/nonat project/Experiment 1/weights/test/DUTS TEST'
test_folder='D:/nonat project/Experiment 1/weights/test/ECCSD TEST'
#test_folder='D:/nonat project/Experiment 1/weights/test/HKU-IS TEST'
#test_folder='D:/nonat project/Experiment 1/weights/test/PASCAL TEST'
#test_folder='D:/nonat project/Experiment 1/weights/test/SOD TEST'

#output_path='D:/nonat project/Experiment 1/Experiment 1 Predictions/DUTS OMRON Saliency Map Prediction/'
#output_path='D:/nonat project/Experiment 1/Experiment 1 Predictions/DUTS Saliency Map Prediction/'
output_path='D:/nonat project/Experiment 1/Experiment 1 Predictions/ECCSD Saliency Map Prediction/'
#output_path='D:/nonat project/Experiment 1/Experiment 1 Predictions/HKU-IS Saliency Map Prediction/'
#output_path='D:/nonat project/Experiment 1/Experiment 1 Predictions/PASCAL Saliency Map Prediction/'
#output_path='D:/nonat project/Experiment 1/Experiment 1 Predictions/SOD Saliency Map Prediction/'
def main(config):

    if config.mode == 'train':
        tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
        print("data_dir + tra_image_dir + '*' + image_ext: ", data_dir + tra_image_dir + '*' + image_ext)

        tra_lbl_name_list = []
        for img_path in tra_img_name_list:
        	img_name = img_path.split("\\")[-1]
        	aaa = img_name.split(".")
        	bbb = aaa[0:-1]
        	imidx = bbb[0]
        	for i in range(1,len(bbb)):
        		imidx = imidx + "." + bbb[i]
        	tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

        print("---")
        print("train images: ", len(tra_img_name_list))
        print("train labels: ", len(tra_lbl_name_list))
        print("---")

        train_num = len(tra_img_name_list)
        salobj_dataset = SalObjDataset(
            img_name_list=tra_img_name_list,
            lbl_name_list=tra_lbl_name_list,
            transform=transforms.Compose([
                RescaleT(256),
                RandomCrop(224),
                ToTensorLab(flag=0)]),
        		category="train",
        		enableInpaintAug=enableInpaintAug)

    test_img_name_list = glob.glob(data_dir + test_image_dir + '*' + image_ext)
    print("data_dir + test_image_dir + '*' + image_ext: ", data_dir + test_image_dir + '*' + image_ext)
    test_lbl_name_list = []
    for img_path in test_img_name_list:
    	img_name = img_path.split("\\")[-1]
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
    if config.mode == 'train':
        train_loader = DataLoader(salobj_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)
        if config.val:
        	val_loader = DataLoader(salobj_dataset_test, batch_size=config.batch_size_val, shuffle=True, num_workers=1)
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_fold, run)): run += 1
        os.mkdir("%s/run-%d" % (config.save_fold, run))
        os.mkdir("%s/run-%d/logs" % (config.save_fold, run))
        # os.mkdir("%s/run-%d/images" % (config.save_fold, run))
        os.mkdir("%s/run-%d/models" % (config.save_fold, run))
        config.save_fold = "%s/run-%d" % (config.save_fold, run)
        if config.val:
            train = Solver(train_loader, val_loader, None, config)
        else:
            train = Solver(train_loader, None, None, config)
        train.train(100)
    elif config.mode == 'test':
        test_loader = DataLoader(salobj_dataset_test, batch_size=batch_size_val, shuffle=True, num_workers=1)
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        test = Solver(None, None, test_loader, config)
        test.test(100,output_path, use_crf=config.use_crf)
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    data_root = os.path.join(os.path.expanduser('~'), 'data')

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=256)  # 256
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--clip_gradient', type=float, default=1.0)
    parser.add_argument('--cuda', type=bool, default=False)

    # Training settings
    parser.add_argument('--vgg', type=str, default=vgg_path)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--val', type=bool, default=True)

    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_fold', type=str, default='./results')
    parser.add_argument('--epoch_val', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=batch_size_train)
    parser.add_argument('--batch_size_val', type=int, default=batch_size_val)
    parser.add_argument('--epoch_save', type=int, default=10)
    parser.add_argument('--epoch_show', type=int, default=1)
    parser.add_argument('--pre_trained', type=str, default=None)

    # Testing settings
    parser.add_argument('--model', type=str, default=trained_model)
    parser.add_argument('--test_fold', type=str, default='./results/test')
    parser.add_argument('--use_crf', type=bool, default=False)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--visdom', type=bool, default=False)
    config = parser.parse_args()
    if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    main(config)
