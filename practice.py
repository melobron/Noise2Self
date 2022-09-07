import os
from glob import glob
import numpy as np
import cv2
import torch

from mask import *
from utils import *
import torchvision.transforms as transforms


mask = Masker(width=4, mode='interpolate')

transform = transforms.Compose([transforms.ToTensor()])
img_dir = '../all_datasets/BSD100'
img_path = glob(os.path.join(img_dir, '*.png'))[0]
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.
img = crop(img, 256)
img_tensor = transform(img)
img_tensor = torch.unsqueeze(img_tensor, dim=0)

i = 0
net_input, mask = mask.mask(img, i)


tensor_list = [img*255., net_input*255., mask*255.]
title_list = ['']
plot_tensors(tensor_list, title_list)

