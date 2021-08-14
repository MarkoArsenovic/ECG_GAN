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

from os import system

import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def normalize(value, linear = True):
    if linear:
        return value/2100
    else:
        return (value - 953.4888085135459) /106

def denormalize(value, linear = True):
    if linear:
        return value * 2100
    else:
        return (value * 106) + 953.4888085135459
    
def normalize_tensor(value, linear = True):
    if linear:
        return  value.div(2100)
    else:
        return value.sub(953.4888085135459).div(106)

def denormalize_tensor(value, linear = True):
    if linear:
        return value.mul(2100)
    else:
        return value.mul(106).add(953.4888085135459)

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

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


def imagefolder_loader(path):
    def loader(transform):
        data = datasets.ImageFolder(path, transform=transform)
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size,
                                 num_workers=4)
        return data_loader
    return loader


def sample_data(dataloader, image_size=4):
    transform = transforms.Compose([
        #transforms.Resize(image_size+int(image_size*0.2)+1),
        #transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    print(dataloader)
    print(image_size)
    loader = dataloader(transform)

    return loader


def train(generator, discriminator, init_step, loader, total_iter=600000):
    step = init_step # can be 1 = 8, 2 = 16, 3 = 32, 4 = 64, 5 = 128, 6 = 128

    #data_loader = sample_data(loader, 4 * 2 ** step)
        
    signals = ECGDataset(data_root = args.path, length =  4 * 2 ** step)
    loader = torch.utils.data.DataLoader(signals, batch_size = batch_size, shuffle=True,num_workers = 2)
    
    dataset = iter(loader)
    #total_iter = 600000
    total_iter_remain = total_iter - (total_iter//6)*(step-1)

    pbar = tqdm(range(total_iter_remain))

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    from datetime import datetime
    import os
    date_time = datetime.now()
    post_fix = '%s_%s_%d_%d.txt'%(trial_name, date_time.date(), date_time.hour, date_time.minute)
    log_folder = 'trial_%s_%s_%d_%d_%d'%(trial_name, date_time.date(), date_time.hour, date_time.minute, date_time.second)
    
    os.mkdir(log_folder)
    os.mkdir(log_folder+'/checkpoint')
    os.mkdir(log_folder+'/sample')

    config_file_name = os.path.join(log_folder, 'train_config_'+post_fix)
    config_file = open(config_file_name, 'w')
    config_file.write(str(args))
    config_file.close()

    log_file_name = os.path.join(log_folder, 'train_log_'+post_fix)
    log_file = open(log_file_name, 'w')
    log_file.write('g,d,nll,onehot\n')
    log_file.close()

    from shutil import copy
    copy('train.py', log_folder+'/train_%s.py'%post_fix)
    copy('progan_modules.py', log_folder+'/model_%s.py'%post_fix)

    alpha = 0
    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = one * -1
    iteration = 0

    generator_loss_during_traning = []
    discriminator_loss_during_traning = []

    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, (2/(total_iter//6)) * iteration)

        if iteration > total_iter//6:
            alpha = 0
            iteration = 0
            step += 1

            if step > 6:
                alpha = 1
                step = 6
            #data_loader = sample_data(loader, 4 * 2 ** step)
            
            signals = ECGDataset(data_root = args.path, length =  4 * 2 ** step)
            loader = torch.utils.data.DataLoader(signals, batch_size = batch_size, shuffle=True,num_workers = 2)
            dataset = iter(loader)

        try:
            #print(next(dataset).shape)
            iteration_next_ds = next(dataset)
            real_image = iteration_next_ds[0]
            label = iteration_next_ds[1]            

        except (OSError, StopIteration):
            dataset = iter(loader)
            real_image, label = next(dataset)
        
        
        iteration += 1

        if(len(real_image) != args.batch_size or len(real_image[0]) == 0):
            print("TOP")
            continue
        
        one =  torch.tensor(1, dtype=torch.float).to(device)
        rone = torch.tensor(np.random.rand()*0.1 + 0.9, dtype=torch.float)

        mone = rone * -1
        
        ### 1. train Discriminator
        b_size = real_image.size(0)
        real_image = real_image.to(device)
        label = label.to(device)
        real_predict = discriminator(
            real_image, step=step, alpha=alpha)
        real_predict = real_predict.mean() \
            - 0.001 * (real_predict ** 2).mean()
        real_predict.backward(mone)

        # sample input data: vector for Generator
        gen_z = torch.randn(b_size, input_code_size).to(device)

        fake_image = generator(gen_z, step=step, alpha=alpha)
        fake_predict = discriminator(
            fake_image.detach(), step=step, alpha=alpha)
        fake_predict = fake_predict.mean()
        fake_predict.backward(one)

        ### gradient penalty for D
        eps = torch.rand(b_size, 1, 1).to(device)#, 1).to(device)

        if(step == 6):
            real_image = real_image.data.reshape([128, 1,  160])
        else:
            real_image = real_image.data.reshape([128, 1,  16 * (2 ** (step - 2))])
        
        x_hat = eps * real_image.data + (1 - eps) * fake_image.detach().data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat, step=step, alpha=alpha)
        grad_x_hat = grad(
            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1)
                         .norm(2, dim=1) - 1)**2).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        
        current_grad_loss_vall = grad_penalty.item()
        current_disc_loss_vall = (real_predict - fake_predict).item()
        grad_loss_val += current_grad_loss_vall
        disc_loss_val += current_disc_loss_vall
        
        generator_loss_during_traning.append(current_grad_loss_vall)
        discriminator_loss_during_traning.append(current_disc_loss_vall)

        d_optimizer.step()

        ### 2. train Generator
        if (i + 1) % n_critic == 0:
            generator.zero_grad()
            discriminator.zero_grad()
            
            predict = discriminator(fake_image, step=step, alpha=alpha)

            loss = -predict.mean()
            gen_loss_val += loss.item()


            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator)

        if (i + 1) % 50 == 0 or i==0:
            with torch.no_grad():
                images = g_running(torch.randn(5 * 10, input_code_size).to(device), step=step, alpha=alpha).data.cpu()

                for img in range(4):
                    plt.plot(np.linspace(1, len(images[0][0]), num = len(images[0][0])), denormalize_tensor(images[img][0].mul(106).add(953.4888085135459)))
                    plt.savefig("./"+log_folder+"/sample/log" + str(i+1) + "_"+str(img)+".png")
                    plt.show()

                plt.plot(np.linspace(1, len(generator_loss_during_traning), num = len(generator_loss_during_traning)),generator_loss_during_traning)
                plt.plot(np.linspace(1, len(discriminator_loss_during_traning), num = len(discriminator_loss_during_traning)),discriminator_loss_during_traning)
                plt.savefig("./"+log_folder+"/sample/logLoss" + str(i+1) + ".png")
                plt.show()
                system('clear')
                
                generator_loss_during_traning = []
                discriminator_loss_during_traning = []

        if (i+1) % 500 == 0 or i==0:
            try:
                torch.save(g_running.state_dict(), f'{log_folder}/checkpoint/{str(i + 1).zfill(6)}_g.model')
                torch.save(discriminator.state_dict(), f'{log_folder}/checkpoint/{str(i + 1).zfill(6)}_d.model')
            except:
                pass

        if (i+1)%50 == 0:
            state_msg = (f'{i + 1}; G: {gen_loss_val/(500//n_critic):.3f}; D: {disc_loss_val/500:.3f};'
                f' Grad: {grad_loss_val/500:.3f}; Alpha: {alpha:.3f}')
            
            log_file = open(log_file_name, 'a+')
            new_line = "%.5f,%.5f\n"%(gen_loss_val/(500//n_critic), disc_loss_val/500)
            log_file.write(new_line)
            log_file.close()

            disc_loss_val = 0
            gen_loss_val = 0
            grad_loss_val = 0

            print(state_msg)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Progressive GAN, during training, the model will learn to generate  images from a low resolution, then progressively getting high resolution ')

    parser.add_argument('--path', type=str, help='path of specified dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--trial_name', type=str, default="test1", help='a brief description of the training trial')
    parser.add_argument('--gpu_id', type=int, default=0, help='0 is the first gpu, 1 is the second gpu, etc.')
    parser.add_argument('--lrD', type=float, default=0.001, help='learning rate, default is 1e-3, usually dont need to change it, you can try make it bigger, such as 2e-3')
    parser.add_argument('--lrG', type=float, default=0.001, help='learning rate, default is 1e-3, usually dont need to change it, you can try make it bigger, such as 2e-3')
    parser.add_argument('--z_dim', type=int, default=128, help='the initial latent vector\'s dimension, can be smaller such as 64, if the dataset is not diverse')
    parser.add_argument('--channel', type=int, default=128, help='determines how big the model is, smaller value means faster training, but less capacity of the model')
    parser.add_argument('--batch_size', type=int, default=4, help='how many images to train together at one iteration')
    parser.add_argument('--n_critic', type=int, default=1, help='train Dhow many times while train G 1 time')
    parser.add_argument('--init_step', type=int, default=1, help='start from what resolution, 1 means 8x8 resolution, 2 means 16x16 resolution, ..., 6 means 256x256 resolution')
    parser.add_argument('--total_iter', type=int, default=300000, help='how many iterations to train in total, the value is in assumption that init step is 1')
    parser.add_argument('--pixel_norm', default=False, action="store_true", help='a normalization method inside the model, you can try use it or not depends on the dataset')
    parser.add_argument('--tanh', default=False, action="store_true", help='an output non-linearity on the output of Generator, you can try use it or not depends on the dataset')
    
    args = parser.parse_args()

    trial_name = args.trial_name
    device = torch.device("cpu")
    input_code_size = args.z_dim
    batch_size = args.batch_size
    n_critic = args.n_critic

    print(args.channel, input_code_size, args.pixel_norm, args.tanh)
    generator = Generator(in_channel=args.channel, input_code_dim=input_code_size, pixel_norm=args.pixel_norm, tanh=args.tanh).to(device)
    discriminator = Discriminator(feat_dim=args.channel).to(device)
    g_running = Generator(in_channel=args.channel, input_code_dim=input_code_size, pixel_norm=args.pixel_norm, tanh=args.tanh).to(device)
    
    ## you can directly load a pretrained model here
    #generator.load_state_dict(torch.load('/home/panonit/Documents/ECG_GAN/Filip/Test/Progressive-GAN-pytorch/trial_experiment-1_2021-08-13_9_27_40/checkpoint/005000_g.model'))
    #g_running.load_state_dict(torch.load('checkpoint/150000_g.model'))
    #discriminator.load_state_dict(torch.load('checkpoint/150000_d.model'))
    
    g_running.train(False)

    g_optimizer = optim.Adam(generator.parameters(), lr=args.lrG, betas=(0.0, 0.99))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lrD, betas=(0.0, 0.99))

    accumulate(g_running, generator, 0)

    signals = ECGDataset(data_root = args.path, length = 4)
    loader = torch.utils.data.DataLoader(signals, batch_size = batch_size, shuffle=True,num_workers = 2)

    train(generator, discriminator, args.init_step, loader, args.total_iter)
