import os
import torch
from progan_modules256 import Generator

import matplotlib.pyplot as plt
import numpy as np

from train256 import ECGDataset, denormalize, denormalize_tensor

latent_vector = torch.randn(128, 128)
input_dimansion_type = [125, 125, 100, 75, 75]

import os
import shutil


for ecg_class in range(5):
    
    if ecg_class < 4:
        continue
    print()
    print("Class ", str(ecg_class))
    print()
    model = Generator(in_channel=128, input_code_dim = input_dimansion_type[ecg_class], pixel_norm=False, tanh=True)
    
    #model.load_state_dict(torch.load('/home/panonit/Documents/ECG_GAN/Filip/Test/Progressive-GAN-pytorch/trial_experiment-1_2021-08-27_12_19_55/checkpoint/001500_g.model')) #all
    
    #model.load_state_dict(torch.load('/home/panonit/Documents/ECG_GAN/Filip/Test/Progressive-GAN-pytorch/trial_experiment-1_2021-08-27_13_4_23/checkpoint/008000_g.model')) #/dataset256/0'
    
    
    
    
    
    """
    with torch.no_grad():
        fake = model(latent_vector).cpu()
    signals = ECGDataset('./dataset', length = 4)
    
    """
    
    #signals = ECGDataset('/home/panonit/Documents/ECG_GAN/Filip/Test/Progressive-GAN-pytorch/dataset256', length = 256) #OC/256/2/', length = 256)
    
    counter  = 0
    
    
    
    model.eval()
    

    
    #os.mkdir('/home/panonit/Documents/ECG_GAN/Filip/Test/Progressive-GAN-pytorch/generatedDS/CheckPoints/dsGenerated/')
    os.mkdir('/home/panonit/Documents/ECG_GAN/Filip/Test/Progressive-GAN-pytorch/generatedDS/CheckPoints/dsGenerated/' + str(ecg_class))
    
    file_index = 0
    
    with torch.no_grad():
        #model.eval()
        for i in range(64):
            print(i)
            gen_z = torch.randn(128, input_dimansion_type[ecg_class])
            model.load_state_dict(torch.load('/home/panonit/Documents/ECG_GAN/Filip/Test/Progressive-GAN-pytorch/generatedDS/CheckPoints/' + str(ecg_class) +'_g.model'))
            fake = model(gen_z, 6, 1).data.cpu()
        
            for fake_image in fake:
                #lowest_std = 9999999999
                #closest_signal = []
        
                #print(fake_image)
                #print(denormalize(fake_image[0], True))
                #plt.plot(np.linspace(1, len(fake_image[0]), num = len(fake_image[0])), denormalize(fake_image[0], True))#.mul(106).add(953.4888085135459))
                #plt.plot(np.linspace(1, len(fake_image[0]), num = len(fake_image[0])), denormalize_tensor(fake_image[0], True))
                counter += 1
                #if counter > 1000:
                #    break
                context = ""
                for info_ds in denormalize_tensor(fake_image[0], True):
                		context += str(info_ds.item()) + "\n"
                		with open('/home/panonit/Documents/ECG_GAN/Filip/Test/Progressive-GAN-pytorch/generatedDS/CheckPoints/dsGenerated/'+ str(ecg_class) +'/'+str(file_index)+'.txt', 'w+') as file:
                			file.write(context)
                file_index += 1
     
    #plt.ylim([775, 1325])
    #plt.show()
    





"""
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
    """