from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
import numpy as np
import torch
import imageio
import torchvision.transforms as transforms
import scipy.io
import random, json
import cv2

class BSDS_RCFLoader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(self.root, 'train_val_all.lst')
            
        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        if self.split == "train":
            img_lb_file = self.filelist[index].strip("\n").split(" ")
            img_file=img_lb_file[0]
            label_list=[]
            for i_label in range(1,len(img_lb_file)):
                lb = scipy.io.loadmat(join(self.root,img_lb_file[i_label]))
                lb=np.asarray(lb['edge_gt'])
                label = torch.from_numpy(lb)
                label = label[1:label.size(0), 1:label.size(1)]
                label = label.float()
                label_list.append(label.unsqueeze(0))
            labels=torch.cat(label_list,0)
            lb_mean=labels.mean(dim=0).unsqueeze(0)
            lb_std=labels.std(dim=0).unsqueeze(0)
            lb_index=random.randint(2,len(img_lb_file))-1
            lb_file=img_lb_file[lb_index]
            
        else:
            img_file = self.filelist[index].rstrip()

        img = imageio.imread(join(self.root,img_file))
        img = transforms.ToTensor()(img)
        img = img[:, 1:img.size(1), 1:img.size(2)]
        img = img.float()

        
        if self.split == "train":
            
            lb = scipy.io.loadmat(join(self.root,lb_file))
            lb=np.asarray(lb['edge_gt'])
            label = torch.from_numpy(lb)
            label = label[1:label.size(0), 1:label.size(1)]
            label = label.unsqueeze(0)
            label = label.float()
                
            return img, label,lb_mean,lb_std
            
        else:
            return img, None

class DMRIRloader(data.Dataset):
    def __init__(self,
                 data_root="C:/dataset/DMR-IR",
                 test_data="DMRIR",
                 test_list='test_pair.lst',
                 mean_bgr=[60.939, 72.779, 60.68, 137.86]
                 ):
        self.data_root = data_root
        self.test_data = test_data
        self.test_list = test_list
        self.images_name = []
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        self.mean_bgr = mean_bgr if len(mean_bgr) == 3 else mean_bgr[:3]
        self.data_index = self._build_index()
    def _build_index(self):
        sample_indices = []

        if not self.test_list:
            raise ValueError(
                f"Test list not provided for dataset: {self.test_data}")

        list_name = os.path.join(self.data_root, self.test_list)
        with open(list_name) as f:
            files = json.load(f)
        for pair in files:
            tmp_img = pair[0]
            tmp_gt = pair[1]
            a = tmp_img[:-3]+"png"
            a = a.replace("npy", 'png')
            name = os.path.basename(a)
            name, ext = os.path.splitext(name)
            sample_indices.append(
                (a,tmp_gt,))
            self.images_name.append(name)

        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        # image_path, label_path = self.data_index[idx]
        if self.data_index[1] is None:
            image_path = self.data_index[0][idx] if len(self.data_index[0]) > 1 else self.data_index[0][idx - 1]
        else:
            image_path = self.data_index[idx][0]
        label_path = None #  self.data_index[idx][1]
        img_name = os.path.basename(image_path)

        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = None
        im_shape = [image.shape[0], image.shape[1]]
        image = self.transform(image)
        label = False if image_path.find("healthy") == -1 else True
        # if label true it is healthy
        # return dict(images=image, labels=label, file_names=file_name, image_shape=im_shape)
        return image, label

    # def transform(self, img, gt):
    #     # Make sure images and labels are divisible by 2^4=16
    #
    #     img = np.array(img, dtype=np.float32)
    #     img -= self.mean_bgr
    #     img = img.transpose((2, 0, 1)) # BGR to RGB
    #     img = torch.from_numpy(img.copy()).float()
    #
    #     gt = np.zeros((img.shape[:2]))
    #     gt = torch.from_numpy(np.array([gt])).float()
    #
    #     return img, gt

