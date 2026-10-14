import argparse
import os
from glob import glob

import imageio
import numpy as np
from torch.utils.data import Dataset
import imutils as imutils
import matplotlib.pyplot as plt

# from PIL import Image

def sample(x, ratio, phase="train"):
    n = int(x*ratio)
    interval = int(x/n)
    full_index = list(range(x))

    train_index  = list(range(0, x, interval))
    test_inex = list(set(full_index).symmetric_difference(set(train_index)))

    assert len(train_index)+len(test_inex) == len(full_index), "The length of the partition is not equal"

    if phase == "train":
        return train_index
    else:
        return test_inex



def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img


class MultimodalDamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, crop_size, type='train', data_loader=img_loader, ext=None, do_sample=True, ratio=0.1):
        self.dataset_path = dataset_path
        self.pre = [] 
        self.post = [] 
        self.label = [] 
        if ext is None:
            ext= ".png"
        if not do_sample:
            for fold in os.listdir(dataset_path):
                self.pre+=glob(f"{dataset_path}/{fold}/pre/*{ext}")
                self.post+=glob(f"{dataset_path}/{fold}/post/*{ext}")
                self.label+=glob(f"{dataset_path}/{fold}/labels/*{ext}")
        else:
            for fold in os.listdir(dataset_path):
                paths = glob(f"{dataset_path}/{fold}/pre/*{ext}")
                N = len(paths)
                sample_inds = sample(x=N, ratio=ratio, phase=type)
                self.pre+=[glob(f"{dataset_path}/{fold}/pre/*{ext}")[i] for i in sample_inds]
                self.post+=[glob(f"{dataset_path}/{fold}/post/*{ext}")[i] for i in sample_inds]
                self.label+=[glob(f"{dataset_path}/{fold}/labels/*{ext}")[i] for i in sample_inds]


        print(f"Found, {len(self.pre)}, {len(type)} images with sample {sample}", os.listdir(dataset_path))

        assert len(self.pre) == len(self.post) == len(self.label), "Train dataset has no equal length"

        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = self.pre[index]
        post_path = self.post[index] 
        label_path = self.label[index] 
        pre_img = self.loader(pre_path)[:,:,0:3] 
        post_img = self.loader(post_path)  
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        # post_img = np.stack((post_img,)*3, axis=-1)
        clf_label = self.loader(label_path)
        # print(np.unique(clf_label))
        

        if 'train' in self.data_pro_type:
            pre_img, post_img, clf_label = self.__transforms(True, pre_img, post_img, clf_label)
        else:
            pre_img, post_img, clf_label = self.__transforms(False, pre_img, post_img, clf_label)
            clf_label = np.asarray(clf_label)

        clf_label[clf_label == 5] = 1

        loc_label = clf_label.copy()
        loc_label[loc_label == 2] = 1  # changing the classification 
        loc_label[loc_label == 3] = 1
        loc_label[loc_label == 4] = 1
        loc_label[loc_label == 5] = 1

        fname = os.path.splitext(os.path.split(pre_path)[1])[0]

        # data_idx = self.data_list[index]
        fname = os.path.splitext(os.path.split(pre_path)[1])[0]
        return pre_img, post_img, loc_label, clf_label, fname

    def __len__(self):
        return len(self.pre)


class MultimodalDamageAssessmentDatset_Inference(Dataset):
    def __init__(self, dataset_path, data_loader=img_loader, suffix='.tif'):
        self.dataset_path = dataset_path
        self.loader = data_loader
        self.suffix = suffix
        self.pre = []
        self.post = []

        for fold in os.listdir(dataset_path):
            self.pre+=glob(f"{dataset_path}/{fold}/pre/*.png")
            self.post+=glob(f"{dataset_path}/{fold}/post/*.png")
        # self.label = [glob(f"{dataset_path}/{type}/{fold}/label/*.png") for fold in os.listdir(dataset_path)]

    def __transforms(self, pre_img, post_img):
        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img

    def __getitem__(self, index):
        pre_path = self.pre[index]
        post_path = self.post[index] 

        assert len(self.pre) == len(self.post), "Test dataset has no equal length"

        pre_img = self.loader(pre_path)[:,:,0:3] 
        post_img = self.loader(post_path) 
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        # post_img = np.stack((post_img,)*3, axis=-1) 
        
        pre_img, post_img = self.__transforms(pre_img, post_img)
    
        data_idx = self.data_list[index]
        return pre_img, post_img, data_idx
    
    def __len__(self):
        return len(self.pre)