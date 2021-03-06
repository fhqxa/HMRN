#-------------------------------------
# Project: Few-shot Learning Based on Hierarchical Classification via Multi-granularity Relation Networks
# Date: 2021.06.30
# Author: Yuling Su
# All Rights Reserved
#-------------------------------------


import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x

def omniglot_character_folders(): 
  

    data_folder ='./datas/omniglot_resized'  

    coarse_character_folders = [os.path.join(data_folder,family) \
                for family in os.listdir(data_folder)]
    random.seed(1)
    random.shuffle(coarse_character_folders)
    num_train = 38  
    metatrain_coarse_character_folders = coarse_character_folders[:num_train]
    metaval_coarse_character_folders = coarse_character_folders[num_train:]
    
    metatrain_character_folders = [os.path.join(coarse_label,label) \
                for coarse_label in metatrain_coarse_character_folders\
                for label in os.listdir(coarse_label)
                ]
    metaval_character_folders = [os.path.join(coarse_label, label) \
                for coarse_label in metaval_coarse_character_folders\
                for label in os.listdir(coarse_label)
                ]
    random.shuffle(metatrain_character_folders)
    random.shuffle(metaval_character_folders)

    return metatrain_character_folders,metaval_character_folders

class OmniglotTask(object): 
    
    def __init__(self, character_folders, num_classes, train_num,test_num):
      
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders,self.num_classes)
    
        coarse_class_folders = np.unique([self.get_class(x) for x in class_folders])
        coarse_labels = np.array(range(len(coarse_class_folders)))
        coarse_labels = dict(zip(coarse_class_folders, coarse_labels))
     
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
     
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:

            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num+test_num]
 
        self.train_coarse_labels = [coarse_labels[self.get_coarse_class(x)] for x in self.train_roots]
        self.test_coarse_labels = [coarse_labels[self.get_coarse_class(x)] for x in self.test_roots]
        self.train_labels = [labels['/'+self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels['/'+self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])

    def get_coarse_class(self, sample):
        return os.path.join(*sample.split('/')[:-2])


class FewShotDataset(Dataset): 

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform 
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        self.coarse_labels = self.task.train_coarse_labels if self.split == 'train' else self.task.test_coarse_labels
    
    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class Omniglot(FewShotDataset):  

    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('L')
        image = image.resize((28,28), resample=Image.LANCZOS) 
    
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        coarse_label = self.coarse_labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)  
        return image, coarse_label, label

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
   
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):

        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=1, split='train',shuffle=True,rotation=0):

    normalize = transforms.Normalize(mean=[0.92206], std=[0.08426]) 

    dataset = Omniglot(task,split=split,transform=transforms.Compose([Rotate(rotation),transforms.ToTensor(),normalize]))

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num,shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num,shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader

