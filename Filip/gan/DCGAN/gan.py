import os

import torch
import torch.nn as nn

import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchvision.utils import save_image
import torchvision.datasets as dateset
import torchvision.transforms as transforms
from torch.utils.data import Dataset



from models.discriminator import Discriminator
from models.generator import Generator

import random

import gc

from pympler.tracker import SummaryTracker
tracker = SummaryTracker()



manualSeed = 9999
random.seed(manualSeed)
torch.manual_seed(manualSeed)


dataset_path = "./dataset/"
batch_size = 128
epochs = 50
learning_rate_D = 0.00018
learning_rate_G = 0.0002
noise = 32


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

dataloader = torch.utils.data.DataLoader(signals, batch_size = batch_size, shuffle=True,num_workers = 2)

#device = torch.device("cuda:0")
device = torch.device("cpu")


generator = Generator(1, noise).to(device)
discriminator = Discriminator(1).to(device)

criterion = nn.BCELoss()

latent_vector = torch.rand(1, noise, 1, 1, device = device)

optimizer_discriminator = optim.Adam(discriminator.parameters(), lr = learning_rate_D, betas = (0.5, 0.999), amsgrad=True)
optimizer_generator = optim.Adam(generator.parameters(), lr = learning_rate_G, betas = (0.5, 0.999), amsgrad=True)

discriminator_error = []
generator_error = []

label_of_zeros = torch.zeros((batch_size,), device = device) 
label_of_ones = 0.1 * torch.rand((batch_size,), device = device) + 0.9 

fixed_noise = torch.randn(batch_size, noise, 1, device = device)

def discriminator_train_real():
    global error_real_epoche_discriminator, score_real_discriminator

    real_output = discriminator(pass_signals).view(-1)
    error_real_epoche_discriminator = criterion(real_output, label_of_ones)
    error_real_epoche_discriminator.backward()
    score_real_discriminator = real_output.mean().item()

def discriminator_train_fake():
    global fake_signals, error_fake_epoche_discriminator, score_fake_discriminator, error_epoche_discriminator

    noise_generate = torch.randn(batch_size, noise, 1, device = device) 

    fake_signals = generator(noise_generate)
    
    fake_output = discriminator(fake_signals.detach()).view(-1)
    
    error_fake_epoche_discriminator = criterion(fake_output, label_of_zeros)
    error_fake_epoche_discriminator.backward()
    score_fake_discriminator = fake_output.mean().item()
    
    error_epoche_discriminator = error_real_epoche_discriminator + error_fake_epoche_discriminator
    optimizer_discriminator.step()


def generator_train():
    global error_epoche_generator
    
    output = discriminator(fake_signals).view(-1)

    error_epoche_generator = criterion(output, label_of_ones)
    error_epoche_generator.backward()
    score_generator = output.mean().item() 
    optimizer_generator.step()

def summerize_iteration():
    global discriminator_error, generator_error
    
    discriminator_error.append(error_fake_epoche_discriminator + error_real_epoche_discriminator)
    generator_error.append(error_epoche_generator)
    if iteration % 25 == 0:
        print("Epoh [{}/{}] Iteration [{}/{}]D: {:.4f} G: {:.4f}".format(epoch, epochs, iteration, len(dataloader), error_fake_epoche_discriminator + error_real_epoche_discriminator, error_epoche_generator))
      

import matplotlib.pyplot as plt

def summerize_epoch():
    with torch.no_grad():
        fake_signals = generator(fixed_noise).detach().cpu()

    plt.plot(fake_signals[0])
    plt.savefig('0gan{}.png'.format(epoch))
    plt.clf()

    plt.plot(fake_signals[1])
    plt.savefig('1gan{}.png'.format(epoch))
    plt.clf()

    plt.plot(fake_signals[2])
    plt.savefig('2gan{}.png'.format(epoch))
    plt.clf()

    plt.plot(fake_signals[3])
    plt.savefig('3gan{}.png'.format(epoch))
    plt.clf()

    if epoch % 5 == 0:
        torch.save(discriminator.state_dict(), "./discriminator"+str(epoch)+".pth")
        torch.save(generator.state_dict(), "./generator"+str(epoch)+".pth")
        tracker.print_diff()


    gc.collect()


for epoch in range(epochs):
    iteration = 0

    for batch in dataloader:
        iteration += 1
        
        if list(batch.size())[0] != 128:
        	break

        pass_signals =  torch.FloatTensor(batch).reshape(128,160,1)#.reshape(160, 128, 1)

        #pass_signals = batch[0].to(device)
        #if pass_signals.size(0) != batch_size:
        #    continue

        # Train Discriminator
        discriminator.zero_grad()
        discriminator_train_real()
        discriminator_train_fake()

        # Train Generator
        generator.zero_grad()
        generator_train()

        summerize_iteration()

    summerize_epoch()

    # Break if error of generator have a pick
    if epoch > 9 and error_epoche_generator > 30:
        break
    
    
    

torch.save(discriminator.state_dict(), "./discriminator.pth")
torch.save(generator.state_dict(), "./generator.pth")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.title("Error of generator and discriminator")
plt.plot(generator_error, label="G")
plt.plot(discriminator_error, label="D")
plt.xlabel("iteration")
plt.ylabel("error")
plt.legend()
plt.show()
