import os

import torch
import torch.nn as nn

import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchvision.utils import save_image
import torchvision.datasets as dateset
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from train256 import normalize, denormalize, denormalize_tensor


from models.Wdiscriminator import Discriminator
from models.Wgenerator import Generator

import numpy as np

import random
import gc

import pywt


from pympler.tracker import SummaryTracker
tracker = SummaryTracker()

def find_min_max_of_ds(signals):
    minimal = min([min(x) for x in [min(i) for i in signals]])
    maximal = max([max(x) for x in [max(i) for i in signals]])
    #[ -0.3100, 4]
    return [minimal, maximal]

"""
maxf = -10
maxc = -10
maxa = -10
minf = 10
minc = 10
mina = 10
for i in range(len(signals)):
    for j in range(len(signals[i])):
        for x in range(len(signals[i][j])):
            if x > 3 and x < 772 and maxc < signals[i][j][x].item():
                maxc = signals[i][j][x].item()
            if x > 3 and x < 772 and minc > signals[i][j][x].item():
                minc = signals[i][j][x].item()
            if x < 3 and maxf < signals[i][j][x].item():
                maxf = signals[i][j][x].item()
            if x < 3 and minf > signals[i][j][x].item():
                minf = signals[i][j][x].item()
            if x == 772 and maxa < signals[i][j][x].item():
                maxa = signals[i][j][x].item()
            if x == 772 and mina > signals[i][j][x].item():
                mina = signals[i][j][x].item()
print(mina, minc, minf, maxa, maxc, maxf)
3:  0.054087139666080475 -0.3100224435329437 0.203125 0.8493247628211975 0.4396030604839325 0.8125
7:  0.13278326392173767 -2.248842716217041 -0.2881610691547394 0.9991559982299805 2.8222544193267822 0.942307710647583
"""

mina = 0.13
minc = -2.25
minf = -0.29
maxa = 1
maxc = 2.85
maxf = 0.95


def normalize_wf(value, typeOf):
    if typeOf == 'a':
        return (value - mina) / (maxa - mina)
    if typeOf == 'c':
        return (value - minc) / (maxc - minc)
    if typeOf == 'f':
        return (value - minf) / (maxf - minf)
    if typeOf == 'l':
        return value / 4

def denormalize_wf(value,typeOf):
    if typeOf == 'a':
        return (value * (maxa - mina)) + mina
    if typeOf == 'c':
        return (value * (maxc - minc)) + minc
    if typeOf == 'f':
        return (value * (maxf - minf)) + minf
    if typeOf == 'l':
        return value * 4



class ECGDataset(Dataset):
    def __init__(self, data_root, length, wl_scales):
        freqs = []
        labels = []
        coefs = []
        avrs = []
        
        max_counter = 256/length
        for classname in os.listdir(data_root):
            class_folder = os.path.join(data_root, classname)

            for sample in os.listdir(class_folder):
                sample_filepath = os.path.join(class_folder, sample)

                with open(sample_filepath, 'r') as sample_file:
                    counter = 0
                    sum_of_values = 0
                    skip_invalid = True
                    add_sample = True
                    original_signal = []
                    scales_wf = np.array([2 ** x for x in range(wl_scales)])

                    for value in sample_file.read().split('\n'):
                        if value != '': 
                            if (int(value)) < 800 or (int(value)) > 1300:
                                add_sample = False
                                break                            
                            original_signal.append(normalize(int(value)))

                    if add_sample:
                        #print("Add")
                        avg_original_signal = np.average(original_signal)
                        coef, freqswf = pywt.cwt(data=original_signal, scales=scales_wf, wavelet='morl')
                        
                        freqs.append(normalize_wf(freqswf, 'f'))
                        avrs.append(normalize_wf(avg_original_signal, 'a'))
                        coefs.append(normalize_wf(coef, 'c'))
                        labels.append(normalize_wf(int(classname), 'l'))
                        
        freqs = torch.FloatTensor(freqs).reshape((len(freqs), wl_scales))
        labels = torch.FloatTensor(labels).reshape(len(labels), 1)
        coefs = torch.FloatTensor(coefs).reshape(len(coefs), wl_scales * 256)
        avrs = torch.FloatTensor(avrs).reshape(len(avrs),1)
        
        self.samples = torch.cat((freqs, labels, coefs, avrs), 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return [self.samples[idx]]
        #return [self.freqs[idx], self.labels[idx], self.coefs[idx], self.avrs[idx]]



manualSeed = 9999
random.seed(manualSeed)
torch.manual_seed(manualSeed)


dataset_path = "./datasetOC/256/1" #"./dataset256/"
batch_size = 32
epochs = 400
learning_rate_D = 0.00005
learning_rate_G = 0.00005
noise = 64
wl_scales = 7

signals = ECGDataset(data_root = dataset_path, length = 256, wl_scales = wl_scales)


def denormalize_wf_tensor(value):
    wf_scales = 7
    for i in range(len(value)):
        if i < wf_scales:
            value[i] = value[i] * (maxa - mina) + mina
        if i == wf_scales:
            value[i] = value[i] * (maxc - minc) + minc
        if i > wf_scales and i < 1 + wf_scales + wf_scales * 256:
            value[i] = value[i] * (maxf - minf) + minf
        if i == 1 + wf_scales + wf_scales * 256:
            value[i] = value[i] * 4
        return value



import matplotlib.pyplot as plt
import numpy as np
import pywt

dataloader = torch.utils.data.DataLoader(signals, batch_size = batch_size, shuffle=True,num_workers = 2)

#device = torch.device("cuda:0")
device = torch.device("cpu")


generator = Generator(2 + wl_scales + wl_scales * 256).to(device)
discriminator = Discriminator(2 + wl_scales + wl_scales * 256).to(device)

criterion = nn.BCELoss()

latent_vector = torch.rand(1, noise, 1, 1, device = device)

optimizer_discriminator = optim.RMSprop(discriminator.parameters(), lr = learning_rate_D)
optimizer_generator = optim.RMSprop(generator.parameters(), lr = learning_rate_G)

discriminator_error = []
generator_error = []

label_of_zeros = torch.zeros((batch_size,), device = device) 
label_of_ones = 0.1 * torch.rand((batch_size,), device = device) + 0.9 

fixed_noise = torch.randn(batch_size, noise, 1, device = device)

def discriminator_train_real():
    global error_real_epoche_discriminator, score_real_discriminator

    real_output = discriminator(pass_signals).view(-1)
    #print(real_output)
    error_real_epoche_discriminator = torch.mean(real_output)
    error_real_epoche_discriminator.backward()
    score_real_discriminator = real_output.mean().item()
    
    for p in discriminator.parameters():
        p.data.clip_(-0.01, 0.01)

def discriminator_train_fake():
    global fake_signals, error_fake_epoche_discriminator, score_fake_discriminator, error_epoche_discriminator

    noise_generate = torch.randn(batch_size, noise, 1, device = device) 
    #print(noise_generate.shape)
    fake_signals = generator(noise_generate)
    #print(fake_signals.shape)
    fake_output = discriminator(fake_signals.detach()).view(-1)
    
    error_fake_epoche_discriminator = -torch.mean(fake_output)
    error_fake_epoche_discriminator.backward()
    score_fake_discriminator = fake_output.mean().item()
    
    error_epoche_discriminator = error_real_epoche_discriminator + error_fake_epoche_discriminator
    optimizer_discriminator.step()
    
    for p in discriminator.parameters():
        p.data.clip_(-0.01, 0.01)


def generator_train():
    global error_epoche_generator
    
    output = discriminator(fake_signals).view(-1)

    error_epoche_generator = -torch.mean(output)
    error_epoche_generator.backward()
    score_generator = output.mean().item() 
    optimizer_generator.step()

def summerize_iteration():
    global discriminator_error, generator_error
    
    if iteration % 32 == 0:
        discriminator_error.append(error_fake_epoche_discriminator + error_real_epoche_discriminator)
        generator_error.append(error_epoche_generator)
        
    if iteration % 25 == 0:
        print("Epoh [{}/{}] Iteration [{}/{}]D: {:.4f} G: {:.4f}".format(epoch, epochs, iteration, len(dataloader), error_fake_epoche_discriminator + error_real_epoche_discriminator, error_epoche_generator))
      


import matplotlib.pyplot as plt

def recreate_signal(signal_wf_grad):
    signal_wf = denormalize_wf_tensor(signal_wf_grad.detach().numpy())
    scales = np.array([2 ** x for x in range(wl_scales)])
    
    coef = np.array(signal_wf[wl_scales + 1 : -1]).reshape((wl_scales, 256))
    freqs = signal_wf[0:wl_scales]
    
    
    mwf = pywt.ContinuousWavelet('morl').wavefun()
    y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]

    r_sum = np.transpose(np.sum(np.transpose(coef)/ scales ** 0.5, axis=-1))
    reconstructed = (r_sum * (1 / y_0))  + signal_wf[-1].item()
    return reconstructed
    

def summerize_epoch():
    if epoch % 20 == 0:
        with torch.no_grad():
            fake_signals = generator(fixed_noise).detach().cpu()

        plt.plot(denormalize(recreate_signal(fake_signals[0])))
        plt.savefig('gan{}1.png'.format(epoch))
        plt.clf()
    
        plt.plot(denormalize(recreate_signal(fake_signals[1])))
        plt.savefig('gan{}2.png'.format(epoch))
        plt.clf()
    
        plt.plot(denormalize(recreate_signal(fake_signals[2])))
        plt.savefig('gan{}3.png'.format(epoch))
        plt.clf()
    
        plt.plot(denormalize(recreate_signal(fake_signals[3])))
        plt.savefig('gan{}4.png'.format(epoch))
        plt.clf()
        
        plt.plot(denormalize(recreate_signal(fake_signals[5])))
        plt.savefig('gan{}5.png'.format(epoch))
        plt.clf()
        
        torch.save(discriminator.state_dict(), "./discriminator"+str(epoch)+".pth")
        torch.save(generator.state_dict(), "./generator"+str(epoch)+".pth")
        tracker.print_diff()
    

    gc.collect()


for epoch in range(epochs):
    iteration = 0

    for batch in dataloader:
        iteration += 1
        
        if len(batch[0]) != batch_size:
            #print(batch.shape)
            #print(batch[2])
            break

        #pass_signals =  torch.FloatTensor(batch).reshape(128,160,1)#.reshape(160, 128, 1) #2319
        #pass_signals =  torch.FloatTensor(batch).reshape(128, 772)#.reshape(160, 128, 1) #2319
        pass_signals = torch.FloatTensor(batch[0]).reshape(batch_size, 1, wl_scales + 2 + wl_scales * 256) #773
        #pass_signals = batch[0].to(device)
        #if pass_signals.size(0) != batch_size:
        #    continue

        # Train Discriminator
        for i in range(3):
            discriminator.zero_grad()
            discriminator_train_real()
            discriminator_train_fake()

        # Train Generator
        generator.zero_grad()
        generator_train()
        
        summerize_iteration()

    summerize_epoch()

    # Break if error of generator have a pick
    #if epoch > 9 and error_epoche_generator > 30:
    #    break
    
    
    

torch.save(discriminator.state_dict(), "./discriminator.pth")
torch.save(generator.state_dict(), "./generator.pth")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.title("Error of generator and discriminator" + str(batch_size) + " " + str(epochs) + " " +str(learning_rate_D) + " " +str(learning_rate_G) + " " +str(noise))
plt.plot([x.detach().numpy() for x in generator_error], label="G")
plt.plot([x.detach().numpy() for x in discriminator_error], label="D")
plt.xlabel("iteration")
plt.ylabel("error")
plt.legend()
plt.show()
