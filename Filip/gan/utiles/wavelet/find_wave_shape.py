import os

import torch
import torch.nn as nn

import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchvision.utils import save_image
import torchvision.datasets as dateset
import torchvision.transforms as transforms
from torch.utils.data import Dataset



from Model128_D.discriminator import Discriminator
from Model128_D.generator import Generator

import random

import gc

from pympler.tracker import SummaryTracker
tracker = SummaryTracker()



manualSeed = 9999
random.seed(manualSeed)
torch.manual_seed(manualSeed)


dataset_path = "./dataset/"


class ECGDataset(Dataset):
    def __init__(self, data_root):
        self.samples = []

        for classname in os.listdir(data_root):
            class_folder = os.path.join(data_root, classname)

            for sample in os.listdir(class_folder):
                sample_filepath = os.path.join(class_folder, sample)

                with open(sample_filepath, 'r') as sample_file:
                    self.sample = []
                    for value in sample_file.read().split('\n'):
                        if value != '':
                            self.sample.append(int(value))
                    self.samples.append(self.sample)
        self.value = torch.FloatTensor(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.value[idx]

signals = ECGDataset(data_root = dataset_path)


import matplotlib.pyplot as plt
import numpy as np
import pywt

my_singal = np.array(signals[1000]) - np.average(signals[1000])

#print(pywt.wavelist())
#['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'dmey', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'haar', 'mexh', 'morl', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'shan', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']

best_std = 999999
best_wave_shape = 'cgau1'

def findSTDofWaveLetAndInverse(singal, wavelet_shape, plot_differene):
    scales = np.array([2 ** x for x in range(8)])
    coef, freqs = pywt.cwt(data=singal, scales=scales, wavelet=wavelet_shape)

    mwf = pywt.ContinuousWavelet(wavelet_shape).wavefun()
    y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]

    r_sum = np.transpose(np.sum(np.transpose(coef)/ scales ** 0.5, axis=-1))
    reconstructed = r_sum * (1 / y_0)

    if plot_differene:
        plt.plot(freqs)
        plt.show()
        plt.plot(singal)
        plt.plot(reconstructed)
        plt.plot(reconstructed - singal)
        print()
        print(wavelet_shape)
        print(np.std(reconstructed - singal))
        plt.show()
    
    return np.std(reconstructed - singal)

for shape in ['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'shan']:
    std_of_shape = findSTDofWaveLetAndInverse(my_singal, shape, False)
    if std_of_shape < best_std:
        best_std = std_of_shape  
        best_wave_shape = shape

std_of_shape = findSTDofWaveLetAndInverse(my_singal, best_wave_shape, True)

