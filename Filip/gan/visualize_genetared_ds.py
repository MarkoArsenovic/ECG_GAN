import os
import torch
from progan_modules256 import Generator

import matplotlib.pyplot as plt
import numpy as np

from train256 import ECGDataset, denormalize, denormalize_tensor

latent_vector = torch.randn(128, 128)


model = Generator(in_channel=128, input_code_dim=100, pixel_norm=False, tanh=True)

#model.load_state_dict(torch.load('/home/panonit/Documents/ECG_GAN/Filip/Test/Progressive-GAN-pytorch/trial_experiment-1_2021-08-27_12_19_55/checkpoint/001500_g.model')) #all

#model.load_state_dict(torch.load('/home/panonit/Documents/ECG_GAN/Filip/Test/Progressive-GAN-pytorch/trial_experiment-1_2021-08-27_13_4_23/checkpoint/008000_g.model')) #/dataset256/0'

signals = ECGDataset('./dataset256', length = 256) #OC/256/2/', length = 256)

counter  = 0

gen_z = torch.randn(128, 100)

model.eval()

with torch.no_grad():
    #model.eval()
    model.load_state_dict(torch.load('./trial_experiment-1_2021-09-06_13_30_33/checkpoint/008000_g.model'))
    fake = model(gen_z, 6, 1).data.cpu()

    for fake_image in fake:
        lowest_std = 9999999999
        closest_signal = []
    
        for signal in signals:
            std_value = 0
            for i in range(len(signal[0])):
                #std_value += (denormalize_tensor(signal[0][i], True - denormalize_tensor(fake_image[0], True)[i]) ** 2 
                std_value += (signal[0][i] - fake_image[0][i]) ** 20 
            #print(std_value) 
            if lowest_std > std_value:
                lowest_std = std_value
                closest_signal = signal[0]
               
        print(fake_image)
        #print(denormalize(fake_image[0], True))
        #plt.plot(np.linspace(1, len(fake_image[0]), num = len(fake_image[0])), denormalize(fake_image[0], True))#.mul(106).add(953.4888085135459))
        plt.plot(np.linspace(1, len(fake_image[0]), num = len(fake_image[0])), denormalize_tensor(fake_image[0], True))
        #plt.plot(np.linspace(1, len(fake_image[0]), num = len(fake_image[0])), fake_image[0].mul(106).add(953.4888085135459))
        #print(len(closest_signal))

        plt.plot(np.linspace(1, len(fake_image[0]), num = len(fake_image[0])), [denormalize_tensor(x, True) for x in closest_signal]) #closest_signal.mul(106).add(953.4888085135459)
    
        plt.show()
        counter += 1
        if counter > 0:
            break
        
        
from config import confuguration
import os
import shutil

# Delete if dateset folder already exist
if os.path.isdir('./dataset'):
	shutil.rmtree('./dataset/')

# Recreate dateset folder
os.mkdir('./dataset')

# make directory of each class
#|   \---gan
#|       +---dataset
#|       |   \---0
#|       |   \---1
#|       |   \---2
#|       |   ...
#|       |   \---N 
for ecg_class in range(len(confuguration.classes)):
	os.mkdir('./dataset/'+str(ecg_class))

X_axis, Y_axis = load_train_test_ECG_dateset()



file_index = 0

# Write files with data
for index_ds in range(len(X_axis)):
	context = ""
	for info_ds in X_axis[index_ds]:
		context += str(info_ds) + "\n"
	with open('./dataset/'+str(Y_axis[index_ds])+'/'+str(file_index)+'.txt', 'w') as file:
		file.write(context)
	file_index += 1
