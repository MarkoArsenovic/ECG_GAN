import os
import torch
import torch.nn as nn
from progan_modules import Generator, Discriminator

import matplotlib.pyplot as plt
import numpy as np

from train import ECGDataset

latent_vector = torch.randn(128, 128)


model = Generator(in_channel=128, input_code_dim=25, pixel_norm=False, tanh=True)

model.load_state_dict(torch.load('/home/panonit/Documents/ECG_GAN/Filip/Test/Progressive-GAN-pytorch/trial_experiment-1_2021-08-14_6_38_14/checkpoint/031000_g.model'))

gen_z = torch.randn(128, 25)

with torch.no_grad():
    fake = model(gen_z, 6, 1).data.cpu()
    


"""
with torch.no_grad():
    fake = model(latent_vector).cpu()
signals = ECGDataset('./dataset', length = 4)

"""

signals = ECGDataset('./dataset', length = 160)

counter  = 0

for fake_image in fake:
    lowest_std = 100000
    closest_signal = []

    for signal in signals:
        std_value = 0
        for i in range(len(signal)):
            std_value = (signal[0][i] - fake_image[0][i]) ** 2
        if lowest_std > std_value:
            lowest_std = std_value
            closest_signal = signal[0]
    plt.plot(np.linspace(1, len(fake_image[0]), num = len(fake_image[0])), fake_image[0].mul(106).add(953.4888085135459))
    plt.plot(np.linspace(1, len(fake_image[0]), num = len(fake_image[0])), [x * 106 + 953.4888085135459 for x in closest_signal]) #closest_signal.mul(106).add(953.4888085135459)

    plt.show()
    counter += 1
    if counter > 10:
        break
