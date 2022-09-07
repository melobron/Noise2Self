import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils import *
from mask import Masker
from models import DnCNN

mnist_train = MNIST('../data/MNIST', download=True, transform=transforms.Compose([transforms.ToTensor()]), train=True)
mnist_test = MNIST('../data/MNIST', download=True, transform=transforms.Compose([transforms.ToTensor()]), train=False)


def add_noise(img):
    return img + torch.randn(img.size()) * 0.4


class SyntheticNoiseDataset(Dataset):
    def __init__(self, data, mode='train'):
        self.mode = mode
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index][0]
        return add_noise(img), img


# Dataset
noisy_mnist_train = SyntheticNoiseDataset(mnist_train, 'train')
noisy_mnist_test = SyntheticNoiseDataset(mnist_test, 'test')

# Add Noise
noisy, clean = noisy_mnist_train[0]
print(noisy.dtype)
# plot_tensors([noisy, clean], ['noisy', 'clean'])

# Masking
masker = Masker(width=4, mode='interpolate')
net_input, mask = masker.mask(noisy.unsqueeze(0), 5)
plot_tensors([mask, noisy[0], net_input[0], net_input[0]-noisy[0]], ['mask', 'noisy', 'net_input', 'diff'])

# # Model
# model = DnCNN(channels=1, num_of_layers=17)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# data_loader = DataLoader(noisy_mnist_train, batch_size=32, shuffle=True)
#
# for i, batch in enumerate(data_loader):
#     noisy_images, clean_images = batch
#
#     net_input, mask = masker.mask(noisy_images, i)
#     net_output = model(net_input)
#
#     loss = criterion(net_output * mask, noisy_images * mask)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if i % 10 == 0:
#         print("Loss (", i, "): \t", round(loss.item(), 4))
#
#     if i == 100:
#         break
#
# # Evaluation
# test_data_loader = DataLoader(noisy_mnist_test, batch_size=32, shuffle=False)
# i, test_batch = next(enumerate(test_data_loader))
# noisy, clean = test_batch
#
# simple_output = model(noisy)
# invariant_output = masker.infer_full_image(noisy, model)
#
# idx = 3
# plot_tensors([clean[idx], noisy[idx], simple_output[idx], invariant_output[idx]],
#             ["gt", "noisy", "Single Pass Inference", "J-Invariant Inference"])

