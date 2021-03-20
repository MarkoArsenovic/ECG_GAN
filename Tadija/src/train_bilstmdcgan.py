from models import DCDiscriminator,BiLSTMDCGenerator
from build_dataset import build_dataset
from helper_functions import initialize,ode_loss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import os
import matplotlib.pyplot as plt


def train_bilstmdcgan(batch_size,train_steps,model_dir,heartbeat_type,use_simulator=True):
	generator=BiLSTMDCGenerator()
	generator.apply(initialize)
	discriminator=DCDiscriminator()
	discriminator.apply(initialize)
	
	data_set=build_dataset(heartbeat_type)
	print("Dataset size: ",len(data_set))
	data_loader=torch.utils.data.DataLoader(data_set,batch_size=batch_size,shuffle=True,num_workers=1)
	 
	cross_entropy_loss=nn.BCELoss()
	if use_simulator:
		mse_loss=nn.MSELoss()
	g_optimizer=optim.Adam(generator.parameters(),lr=0.0002,betas=(0.5,0.999))
	d_optimizer=optim.Adam(discriminator.parameters(),lr=0.0002,betas=(0.5,0.999))

	ce_d_real_hist=[]
	ce_d_fake_hist=[]
	ce_g_fake_hist=[]
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

			predictions=discriminator(batch)
			labels=torch.full((size,1,1),1,device='cpu').float()
			ce_loss_d_real=cross_entropy_loss(predictions,labels)
			ce_loss_d_real.backward()
			
			noise=torch.Tensor(np.random.normal(0,1,(size,100)))
			synthetic_beats=generator(noise)
			predictions=discriminator(synthetic_beats.detach())
			labels=torch.full((size,1,1),0,device='cpu').float()
			ce_loss_d_fake=cross_entropy_loss(predictions,labels)
			ce_loss_d_fake.backward()

			d_optimizer.step()

			generator.zero_grad()
			predictions=discriminator(synthetic_beats)
			labels=torch.full((size,1,1),1,device='cpu').float()
			ce_loss_g_fake=cross_entropy_loss(predictions,labels)

			if use_simulator:
				delta,ode_signal=ode_loss(synthetic_beats,heartbeat_type)
				mse_loss_euler=mse_loss(delta,ode_signal)
				total_g_loss=mse_loss_euler+ce_loss_g_fake
				total_g_loss.backward()
			else:
				ce_loss_g_fake.backward()

			g_optimizer.step()

			print("Step {} done. (BiLSTMDCGAN, heartbeat type {})".format(steps,heartbeat_type))
			ce_d_real_hist.append(ce_loss_d_real.item())
			ce_d_fake_hist.append(ce_loss_d_fake.item())
			ce_g_fake_hist.append(ce_loss_g_fake.item())
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
					plt.plot(ce_d_real_hist,label="ce_loss_d_real")
					plt.plot(ce_d_fake_hist,label="ce_loss_d_fake")
					plt.plot(ce_g_fake_hist,label="ce_loss_g_fake")
					plt.legend()
					plt.subplot(1,2,2)
					plt.plot(euler_hist,label="mse_loss_euler")
					plt.legend()
				else:
					plt.plot(ce_d_real_hist,label="ce_loss_d_real")
					plt.plot(ce_d_fake_hist,label="ce_loss_d_fake")
					plt.plot(ce_g_fake_hist,label="ce_loss_g_fake")
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
	train_bilstmdcgan(200,2000,"GAN\\LSTM\\BiLSTMDCGAN\\00_N","N",use_simulator=False)
