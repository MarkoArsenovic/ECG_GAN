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


from models.Wd import Discriminator512
from models.Wg import Generator512

import numpy as np

import random
import gc

import pywt


from pympler.tracker import SummaryTracker
tracker = SummaryTracker()


class ECGDataset(Dataset):
    def __init__(self, data_root, length):
        self.samples = []
        self.labels = []
        max_counter = 256/length
        for classname in os.listdir(data_root):
            class_folder = os.path.join(data_root, classname)

            for sample in os.listdir(class_folder):
                sample_filepath = os.path.join(class_folder, sample)

                with open(sample_filepath, 'r') as sample_file:
                    self.sample = []
                    counter = 0
                    sum_of_values = 0
                    skip_invalid = True
                    add_sample = True
                    for value in sample_file.read().split('\n'):
                        if value != '':
                            if (int(value)) < 800 or (int(value)) > 1300:
                                add_sample = False
                                break
                            sum_of_values += normalize(int(value), True)
                            counter += 1
                            if (counter >= max_counter):
                                if skip_invalid:
                                    avg_of_values = sum_of_values / int(counter)
                                    self.sample.append(avg_of_values)
                                else:
                                    skip_invalid = True
                                counter -= max_counter
                                sum_of_values = 0
                    if add_sample:
                        self.samples.append(self.sample)
                        self.labels.append(int(classname))
        self.value = torch.FloatTensor(self.samples)
        self.labels = torch.FloatTensor(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return [self.value[idx], self.labels[idx]]




manualSeed = 9999
random.seed(manualSeed)
torch.manual_seed(manualSeed)


dataset_path = "./datasetOC/256/1/" #"./dataset256/"
batch_size = 32
epochs = 50
learning_rate_D = 0.00005
learning_rate_G = 0.00005
noise = 100

signals = ECGDataset(data_root = dataset_path, length = 256)


dataloader = torch.utils.data.DataLoader(signals, batch_size = batch_size, shuffle=True,num_workers = 2)

#device = torch.device("cuda:0")
device = torch.device("cpu")


generator = Generator512(1).to(device)
discriminator = Discriminator512(1).to(device)

#criterion = nn.BCELoss()
criterion = nn.MSELoss()

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
    error_real_epoche_discriminator = -torch.mean(real_output)
    error_real_epoche_discriminator.backward()
    score_real_discriminator = real_output.mean().item()
    
    for p in discriminator.parameters():
        p.data.clip_(-0.01, 0.01)

def discriminator_train_fake():
    global fake_signals, error_fake_epoche_discriminator, score_fake_discriminator, error_epoche_discriminator

    noise_generate = torch.randn(batch_size, noise, 1, device = device) 
    #print(noise_generate.shape)
    #fake_signals = generator(noise_generate).reshape(batch_size, 1, 256) #.reshape(256, 1, 4)
    fake_signals = generator(noise_generate)
    #print(fake_signals.shape)
    fake_signals.reshape(batch_size, 1, 256)
    
    fake_output = discriminator(fake_signals.detach()).view(-1)
    
    error_fake_epoche_discriminator = torch.mean(fake_output)
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
    
    if iteration % 126 == 0:
        discriminator_error.append(error_fake_epoche_discriminator + error_real_epoche_discriminator)
        generator_error.append(error_epoche_generator)
        
    if iteration % 25 == 0:
        print("Epoh [{}/{}] Iteration [{}/{}]D: {:.4f} G: {:.4f}".format(epoch, epochs, iteration, len(dataloader), error_fake_epoche_discriminator + error_real_epoche_discriminator, error_epoche_generator))
   

import matplotlib.pyplot as plt


    
def summerize_epoch():
    #if epoch % 3 == 0:
    with torch.no_grad():
        fake_signals = generator(fixed_noise).detach().cpu()

    print(fake_signals.shape)
    plt.plot(denormalize(fake_signals[0][0]))
    plt.savefig('gan{}1.png'.format(epoch))
    plt.clf()

    plt.plot(denormalize(fake_signals[1][0]))
    plt.savefig('gan{}2.png'.format(epoch))
    plt.clf()

    plt.plot(denormalize(fake_signals[2][0]))
    plt.savefig('gan{}3.png'.format(epoch))
    plt.clf()

    plt.plot(denormalize(fake_signals[3][0]))
    plt.savefig('gan{}4.png'.format(epoch))
    plt.clf()
    
    plt.plot(denormalize(fake_signals[5][0]))
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
        pass_signals = torch.FloatTensor(batch[0]).reshape(batch_size, 1, 256) #773
        #pass_signals = batch[0].to(device)
        #if pass_signals.size(0) != batch_size:
        #    continue

        # Train Discriminator
        for i in range(6):
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

plt.figure(figsize=(10,5))
plt.title("Error of generator and discriminator" + str(batch_size) + " " + str(epochs) + " " +str(learning_rate_D) + " " +str(learning_rate_G) + " " +str(noise))
plt.plot([x.detach().numpy() for x in generator_error], label="G")
plt.plot([x.detach().numpy() for x in discriminator_error], label="D")
plt.xlabel("iteration")
plt.ylabel("error")
plt.legend()
plt.show()
