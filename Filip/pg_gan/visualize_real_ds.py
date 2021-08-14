from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import random

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from progan_modules import Generator, Discriminator

import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from train import normalize_tensor, normalize

"""
class ECGDataset(Dataset):
    def __init__(self, data_root, length):
        self.samples = []
        self.labels = []
        max_counter = 160/length
        for classname in os.listdir(data_root):
            class_folder = os.path.join(data_root, classname)

            for sample in os.listdir(class_folder):
                sample_filepath = os.path.join(class_folder, sample)

                with open(sample_filepath, 'r') as sample_file:
                    self.sample = []
                    counter = 0
                    for value in sample_file.read().split('\n'):
                        if value != '':
                            if counter >= max_counter-1:
                                self.sample.append(int(value))
                                counter = 0
                            else:
                                counter += 1
                    self.samples.append(self.sample)
                    self.labels.append(int(classname))
        self.value = torch.FloatTensor(self.samples)
        self.labels = torch.FloatTensor(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return [self.value[idx], self.labels[idx]]
"""

"""
class ECGDataset(Dataset):
    def __init__(self, data_root, length):
        self.samples = []
        self.labels = []
        print()
        max_counter = 160/length
        for classname in os.listdir(data_root):
            class_folder = os.path.join(data_root, classname)

            for sample in os.listdir(class_folder):
                sample_filepath = os.path.join(class_folder, sample)

                with open(sample_filepath, 'r') as sample_file:
                    self.sample = []
                    counter = 0
                    sum_of_values = 0
                    for value in sample_file.read().split('\n'):
                        if value != '':
                            sum_of_values += int(value)
                            counter += 1
                            if (counter >= max_counter):
                                avg_of_values = sum_of_values / int(counter)
                                self.sample.append(avg_of_values)
                                counter -= max_counter
                                sum_of_values = 0
                                
                    self.samples.append(self.sample)
                    self.labels.append(int(classname))
        self.value = torch.FloatTensor(self.samples)
        self.labels = torch.FloatTensor(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return [self.value[idx], self.labels[idx]]
"""


class ECGDataset(Dataset):
    def __init__(self, data_root, length):
        self.samples = []
        self.labels = []
        max_counter = 160/length
        for classname in os.listdir(data_root):
            class_folder = os.path.join(data_root, classname)

            for sample in os.listdir(class_folder):
                sample_filepath = os.path.join(class_folder, sample)

                with open(sample_filepath, 'r') as sample_file:
                    self.sample = []
                    counter = 0
                    sum_of_values = 0
                    for value in sample_file.read().split('\n'):
                        if value != '':
                            sum_of_values += normalize(int(value))
                            counter += 1
                            if (counter >= max_counter):
                                avg_of_values = sum_of_values / int(counter)
                                self.sample.append(avg_of_values)
                                counter -= max_counter
                                sum_of_values = 0
                                
                    self.samples.append(self.sample)
                    self.labels.append(int(classname))
        self.value = torch.FloatTensor(self.samples)
        self.labels = torch.FloatTensor(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return [self.value[idx], self.labels[idx]]
    
signals_16 = ECGDataset(data_root = './dataset', length = 16)
signals_32 = ECGDataset(data_root = './dataset', length = 32)
signals_64 = ECGDataset(data_root = './dataset', length = 64)
signals_128 = ECGDataset(data_root = './dataset', length = 128)
signals_160 = ECGDataset(data_root = './dataset', length = 160)

loader_16 = torch.utils.data.DataLoader(signals_16, batch_size = 5, shuffle=True,num_workers = 2)
loader_32 = torch.utils.data.DataLoader(signals_32, batch_size = 5, shuffle=True,num_workers = 2)
loader_64 = torch.utils.data.DataLoader(signals_64, batch_size = 5, shuffle=True,num_workers = 2)
loader_128 = torch.utils.data.DataLoader(signals_128, batch_size = 5, shuffle=True,num_workers = 2)
loader_160 = torch.utils.data.DataLoader(signals_160, batch_size = 5, shuffle=True,num_workers = 2)

dataset_16 = iter(loader_16)
dataset_32 = iter(loader_32)
dataset_64 = iter(loader_64)
dataset_128 = iter(loader_128)
dataset_160 = iter(loader_160)

real_image_16, label_16 = next(dataset_16)
real_image_32, label_32 = next(dataset_32)
real_image_64, label_64 = next(dataset_64)
real_image_128, label_128 = next(dataset_128)
real_image_160, label_160 = next(dataset_160)

print(real_image_16[0])


def plot_signals(signals):
    for i in range(len(signals)):
        plt.plot(np.linspace(1, len(signals[i]), num = len(signals[i])), normalize_tensor(signals[i]))
    plt.show()

plot_signals(real_image_16)
plot_signals(real_image_32)
plot_signals(real_image_64)
plot_signals(real_image_128)
plot_signals(real_image_160)
