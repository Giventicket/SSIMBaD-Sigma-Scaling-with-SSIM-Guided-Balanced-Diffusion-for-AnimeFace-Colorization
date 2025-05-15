import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils.image import warp_image

from torch.utils.data import DataLoader
import random

class Anime(Dataset):

    def __init__(
        self,
        reference_path, # reference
        condition_path, # sketch
        gt_path, # gt_sketch
        size=512,
        torch_dtype='float32',
        distored_condition=True,
    ):
        self.reference_path = reference_path
        self.condition_path = condition_path
        self.gt_path = gt_path
        
        self.tf_ref = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        self.tf_con = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        self.ref_filenames = os.listdir(self.reference_path)
        self.con_filenames = os.listdir(self.condition_path)
        self.gt_filenames = os.listdir(self.gt_path)
        
        self.ref_filenames.sort()
        self.con_filenames.sort()
        self.gt_filenames.sort()
        
        self.size = size
        self.dtype = getattr(torch, torch_dtype)
        self.distored_condition = distored_condition

    def __getitem__(self, index):
        assert self.gt_filenames[index] == self.con_filenames[index]

        ret = {}
        with Image.open(os.path.join(self.reference_path, self.ref_filenames[index])) as img_ref:
            img_reference = img_ref.convert('RGB')
            
        with Image.open(os.path.join(self.gt_path, self.gt_filenames[index])) as img_gt:
            img_gt = img_gt.convert('RGB')
        
        with Image.open(os.path.join(self.condition_path, self.con_filenames[index])) as img_con:
            img_condition = img_con.convert('L')
            
        if self.distored_condition:
            img_distorted = warp_image(img_reference)
        else:
            img_distorted = img_reference

        ret['reference'] = self.tf_ref(img_reference).to(self.dtype)
        ret['condition'] = self.tf_con(img_condition).to(self.dtype)
        ret['distorted'] = self.tf_ref(img_distorted).to(self.dtype)
        ret['name'] = self.con_filenames[index]
        ret['gt'] = self.tf_ref(img_gt).to(self.dtype)
        return ret

    def __len__(self):
        return len(self.con_filenames)