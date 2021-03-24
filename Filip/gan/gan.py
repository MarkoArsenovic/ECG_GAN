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

'''
my_singal = np.array(signals[0]) - np.average(signals[0])
print(pywt.wavelist())
#['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']:
#['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'dmey', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'haar', 'mexh', 'morl', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'shan', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']
for i in ['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'shan']:
    plt.plot(my_singal)
    #plt.show()
    print()
    print(i)
    scales = np.array([2 ** x for x in range(7)])
    coef, freqs = pywt.cwt(data=my_singal, scales=scales, wavelet=i)


    mwf = pywt.ContinuousWavelet(i).wavefun()
    y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]

    r_sum = np.transpose(np.sum(np.transpose(coef)/ scales ** 0.5, axis=-1))
    reconstructed = r_sum * (1 / y_0)

    plt.plot(reconstructed)
    plt.plot(reconstructed - my_singal)
    print(np.std(reconstructed - my_singal))

    plt.show()

(root = dataset_path,
                            transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
'''

dataloader = torch.utils.data.DataLoader(images, batch_size = batch_size, shuffle=True,num_workers = 2)

#device = torch.device("cuda:0")
device = torch.device("cpu")


generator = Generator(1, noise).to(device)
discriminator = Discriminator(1).to(device)

criterion = nn.BCELoss()

latent_vector = torch.rand(1, noise, 1, 1, device = device)


'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

discriminator.apply(weights_init)
generator.apply(weights_init)
'''

optimizer_discriminator = optim.Adam(discriminator.parameters(), lr = learning_rate_D, betas = (0.5, 0.999), amsgrad=True)
optimizer_generator = optim.Adam(generator.parameters(), lr = learning_rate_G, betas = (0.5, 0.999), amsgrad=True)

discriminator_error = []
generator_error = []

label_of_zeros = torch.zeros((batch_size,), device = device) 
label_of_ones = 0.1 * torch.rand((batch_size,), device = device) + 0.9 

fixed_noise = torch.randn(batch_size, noise, 1, device = device)

def discriminator_train_real():
    global error_real_epoche_discriminator, score_real_discriminator

    real_output = discriminator(pass_images).view(-1)
    error_real_epoche_discriminator = criterion(real_output, label_of_ones)
    error_real_epoche_discriminator.backward()
    score_real_discriminator = real_output.mean().item()

def discriminator_train_fake():
    global fake_images, error_fake_epoche_discriminator, score_fake_discriminator, error_epoche_discriminator

    noise_generate = torch.randn(batch_size, noise, 1, device = device) 

    fake_images = generator(noise_generate)
    
    fake_output = discriminator(fake_images.detach()).view(-1)
    
    error_fake_epoche_discriminator = criterion(fake_output, label_of_zeros)
    error_fake_epoche_discriminator.backward()
    score_fake_discriminator = fake_output.mean().item()
    
    error_epoche_discriminator = error_real_epoche_discriminator + error_fake_epoche_discriminator
    optimizer_discriminator.step()


def generator_train():
    global error_epoche_generator
    
    output = discriminator(fake_images).view(-1)

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
        fake_images = generator(fixed_noise).detach().cpu()

    plt.plot(fake_images[0])
    plt.savefig('0gan{}.png'.format(epoch))
    plt.clf()

    plt.plot(fake_images[1])
    plt.savefig('1gan{}.png'.format(epoch))
    plt.clf()

    plt.plot(fake_images[2])
    plt.savefig('2gan{}.png'.format(epoch))
    plt.clf()

    plt.plot(fake_images[3])
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

        pass_images =  torch.FloatTensor(batch).reshape(128,160,1)#.reshape(160, 128, 1)

        #pass_images = batch[0].to(device)
        #if pass_images.size(0) != batch_size:
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
