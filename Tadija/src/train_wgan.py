from models import WGenerator,WDiscriminator,W8Generator,W8Discriminator,W9Generator,W9Discriminator
from build_dataset import build_dataset
from helper_functions import initialize,ode_loss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import os
import matplotlib.pyplot as plt


def train_wgan(batch_size,train_steps,model_dir,heartbeat_type,use_simulator=True):
	#generator=WGenerator()
	generator=W8Generator()
	#generator=W9Generator()
	generator.apply(initialize)
	#discriminator=WDiscriminator()
	discriminator=W8Discriminator()
	#discriminator=W9Discriminator()
	discriminator.apply(initialize)
	
	data_set=build_dataset(heartbeat_type)
	print("Dataset size: ",len(data_set))
	data_loader=torch.utils.data.DataLoader(data_set,batch_size=batch_size,shuffle=True,num_workers=1)
	 
	g_optimizer=optim.RMSprop(generator.parameters(),lr=0.00005)
	d_optimizer=optim.RMSprop(discriminator.parameters(),lr=0.00005)

	weight_cliping_limit=0.01

	d_real_hist=[]
	d_fake_hist=[]
	g_fake_hist=[]
	if use_simulator:
		euler_hist=[]
	
	val_noise=torch.Tensor(np.random.normal(0,1,(4,100)))
	steps=0
	while True:
		for i,batch in enumerate(data_loader):
			if steps>=train_steps:
				break
			steps+=1

			batch=batch.float()
			size=batch.shape[0]
			
			discriminator.zero_grad()

			for p in discriminator.parameters():
				p.data.clamp_(-weight_cliping_limit,weight_cliping_limit)

			predictions=discriminator(batch)
			loss_d_real=-torch.mean(predictions)
			loss_d_real.backward()
			
			noise=torch.Tensor(np.random.normal(0,1,(size,100)))
			synthetic_beats=generator(noise)
			predictions=discriminator(synthetic_beats.detach())
			loss_d_fake=torch.mean(predictions)
			loss_d_fake.backward()
			
			d_optimizer.step()

			generator.zero_grad()
			predictions=discriminator(synthetic_beats)
			loss_g_fake=-torch.mean(predictions)
			
			if use_simulator:
				delta,ode_signal=ode_loss(synthetic_beats,heartbeat_type)
				mse_loss_euler=mse_loss(delta,ode_signal)
				total_g_loss=mse_loss_euler+loss_g_fake
				total_g_loss.backward()
			else:
				loss_g_fake.backward()

			g_optimizer.step()

			print("Step {} done. (WGAN, heartbeat type {})".format(steps,heartbeat_type))
			d_real_hist.append(loss_d_real.item())
			d_fake_hist.append(loss_d_fake.item())
			g_fake_hist.append(loss_g_fake.item())
			if use_simulator:
				euler_hist.append(mse_loss_euler.item())

			if steps%50==0:
				with torch.no_grad():
					output_g=generator(val_noise)
					plt.title("Generated heartbeats {}".format(steps))
					for p in range(min(4,len(batch))):
						plt.subplot(2,2,p+1)
						plt.plot(output_g[p].detach().numpy(),label="synthetic beat")
						plt.plot(batch[p].detach().numpy(),label="real beat")
						plt.legend()
					plt.savefig(os.path.join(model_dir,'synthetic_beats_checkpoint_{}'.format(steps)))
					plt.close()

				plt.title("Loss {}".format(steps))
				if use_simulator:
					plt.subplot(1,2,1)
					plt.plot(d_real_hist,label="loss_d_real")
					plt.plot(d_fake_hist,label="loss_d_fake")
					plt.plot(g_fake_hist,label="loss_g_fake")
					plt.legend()
					plt.subplot(1,2,2)
					plt.plot(euler_hist,label="mse_loss_euler")
					plt.legend()
				else:
					plt.plot(d_real_hist,label="loss_d_real")
					plt.plot(d_fake_hist,label="loss_d_fake")
					plt.plot(g_fake_hist,label="loss_g_fake")
					plt.legend()
				plt.savefig(os.path.join(model_dir,'loss_checkpoint_{}'.format(steps)))
				plt.close()

			
			if steps%250==0:
				torch.save({
					'generator_state_dict': generator.state_dict(),
					'discriminator_state_dict': discriminator.state_dict(),
					'optimizer_g_state_dict': g_optimizer.state_dict(),
					'optimizer_d_state_dict': d_optimizer.state_dict(),
					},os.path.join(model_dir,'training_checkpoint_{}'.format(steps)))


		if steps>=train_steps:
			break
	torch.save({
		'generator_state_dict': generator.state_dict(),
		'discriminator_state_dict': discriminator.state_dict(),
		'optimizer_g_state_dict': g_optimizer.state_dict(),
		'optimizer_d_state_dict': d_optimizer.state_dict(),
	},os.path.join(model_dir,'trained_gan'))


if __name__=="__main__":
	#train_wgan(200,2000,"GAN\\WGAN\\6_layers\\00_N","N",use_simulator=False)
	#train_wgan(200,2000,"GAN\\WGAN\\6_layers\\01_L","L",use_simulator=False)
	#train_wgan(200,2000,"GAN\\WGAN\\6_layers\\02_R","R",use_simulator=False)
	#train_wgan(200,2000,"GAN\\WGAN\\6_layers\\03_A","A",use_simulator=False)

	#train_wgan(200,5000,"GAN\\WGAN\\9_layers\\00_N","N",use_simulator=False)

	train_wgan(200,4000,"GAN\\WGAN\\8_layers\\00_N","N",use_simulator=False)
