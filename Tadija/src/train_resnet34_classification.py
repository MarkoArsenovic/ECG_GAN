from models import DCGenerator,ResNet34
from build_dataset import build_validation,build_testset
from helper_functions import initialize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import os
import matplotlib.pyplot as plt


def train_resnet34_classification(batch_size,train_steps,model_dir,heartbeat_types,models):
	assert len(heartbeat_types)==len(models)
	generators=[]
	num_b=len(heartbeat_types)

	for i in range(num_b):
		chkp=torch.load(models[i])
		gen=DCGenerator()
		gen.load_state_dict(chkp['generator_state_dict'])
		generators.append(gen)
		'''
		with torch.no_grad():
			noise=torch.Tensor(np.random.normal(0,1,(4,100)))
			beats=generators[i](noise)
			for j in range(4):
				plt.subplot(2,2,j+1);
				plt.plot(beats[j].detach().numpy())
			plt.show()
		'''

	classifier=ResNet34(num_b)
	classifier.apply(initialize)

	cross_entropy_loss=nn.CrossEntropyLoss()
	optimizer=optim.SGD(classifier.parameters(),lr=0.0002)
	
	val_x,val_y=build_validation(heartbeat_types)
	val_x=torch.Tensor(val_x)
	val_y=torch.Tensor(val_y)
	ptr=0

	loss_hist=[]
	accuracy_hist=[]
	validation_hist=[]
	
	steps=0
	while True:
		if steps>=train_steps:
			break
		steps+=1

		heartbeats,classes=None,None
		with torch.no_grad():
			for i in range(num_b):
				noise=torch.Tensor(np.random.normal(0,1,(batch_size//num_b,100)))
				beats=generators[i](noise)
				labels=torch.Tensor([i for _ in range(batch_size//num_b)]).long()
				if heartbeats is None:
					heartbeats=beats
					classes=labels
				else:
					heartbeats=torch.cat((heartbeats,beats),0)
					classes=torch.cat((classes,labels),0)

		for i in range(len(heartbeats)):
			j=np.random.randint(i+1)
			if i!=j:
				heartbeats[i],heartbeats[j]=heartbeats[j],heartbeats[i]
				classes[i],classes[j]=classes[j],classes[i]

		classifier.zero_grad()

		predictions=classifier(heartbeats.detach())
		loss=cross_entropy_loss(predictions,classes)

		loss.backward()
		optimizer.step()

		correct=0
		for i in range(len(classes)):
			k,l=0,classes[i]
			for j in range(num_b):
				if predictions[i][j]>predictions[i][k]:
					k=j
			if k==l:
				correct+=1

		accuracy=correct/len(classes)
		accuracy_hist.append(accuracy)

		with torch.no_grad():
			x,y=val_x[ptr:min(ptr+batch_size,len(val_x))],val_y[ptr:min(ptr+batch_size,len(val_y))]
			ptr+=batch_size
			if ptr>=len(val_x):
				ptr=0
			val_pred=classifier(x)
			correct=0
			for i in range(len(y)):
				k,l=0,0
				for j in range(num_b):
					if val_pred[i][j]>val_pred[i][k]:
						k=j
					if y[i][j]>y[i][l]:
						l=j
				if k==l:
					correct+=1
			validation=correct/len(y)
			validation_hist.append(validation)

		loss_hist.append(loss.item())

		print("Step {} done. Loss={:.4f}, accuracy={:.4f}, validation={:.4f}".format(steps,loss.item(),accuracy,validation))

		if steps%50==0:
			plt.subplot(1,2,1)
			plt.title("Loss {}".format(steps))
			plt.plot(loss_hist)
			plt.subplot(1,2,2)
			plt.title("Accuracy {}".format(steps))
			plt.plot(accuracy_hist,label="training")
			plt.plot(validation_hist,label="validation")
			plt.legend()
			plt.savefig(os.path.join(model_dir,'loss_checkpoint_{}'.format(steps)))
			plt.close()

		if steps%250==0:
			torch.save({
					'classifier_state_dict': classifier.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
				},os.path.join(model_dir,'training_checkpoint_{}'.format(steps)))

	torch.save({
			'classifier_state_dict': classifier.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
		},os.path.join(model_dir,'trained_classifier'))
	
	test_x,test_y=build_testset(heartbeat_types)
	test_x=torch.Tensor(test_x)
	test_y=torch.Tensor(test_y)

	predictions=classifier(test_x)
	confusion_matrix=[[0 for _ in range(num_b)] for _ in range(num_b)]
	correct=0
	for i in range(len(test_y)):
		k,l=0,0
		for j in range(num_b):
			if predictions[i][j]>predictions[i][k]:
				k=j
			if test_y[i][j]>test_y[i][l]:
				l=j
		if k==l:
			correct+=1
		confusion_matrix[l][k]+=1
	accuracy=correct/len(test_y)
	print("Testset accuracy={:.10f}={}/{}".format(accuracy,correct,len(test_y)))
	print("Confusion matrix:")
	for i in range(num_b):
		for j in range(num_b):
			print(confusion_matrix[i][j],end=" ")
		print()


def test_pretrained(model,heartbeat_types):
	num_b=len(heartbeat_types)
	classifier=ResNet34(num_b)
	chkp=torch.load(model)
	classifier.load_state_dict(chkp['classifier_state_dict'])

	test_x,test_y=build_testset(heartbeat_types)
	test_x=torch.Tensor(test_x)
	test_y=torch.Tensor(test_y)

	batch_size=200
	ptr=0
	confusion_matrix=[[0 for _ in range(num_b)] for _ in range(num_b)]
	correct=0

	while ptr<test_x.size(0): 
		x,y=test_x[ptr:min(ptr+batch_size,test_x.size(0))],test_y[ptr:min(ptr+batch_size,test_x.size(0))]
		ptr+=batch_size
		predictions=classifier(x)
		for i in range(len(y)):
			k,l=0,0
			for j in range(num_b):
				if predictions[i][j]>predictions[i][k]:
					k=j
				if y[i][j]>y[i][l]:
					l=j
			if k==l:
				correct+=1
			confusion_matrix[l][k]+=1

	accuracy=correct/len(test_y)
	print("Testset accuracy={:.10f}={}/{}".format(accuracy,correct,len(test_y)))
	print("Confusion matrix:")
	for i in range(num_b):
		for j in range(num_b):
			print(confusion_matrix[i][j],end=" ")
		print()


if __name__=="__main__":
	'''
	train_resnet34_classification(240,3000,"Classification\\WithGAN\\SplittedDataset\\ResNet34",["N","L","R","A","a","J","S","V","F","e","j","E"],[
			"GAN\\DCGAN\\6_layers\\00_N\\training_checkpoint_2000",
			"GAN\\DCGAN\\6_layers\\01_L\\training_checkpoint_2000",
			"GAN\\DCGAN\\6_layers\\02_R\\training_checkpoint_2000",
			"GAN\\DCGAN\\6_layers\\03_A\\training_checkpoint_2000",
			"GAN\\DCGAN\\6_layers\\04_a\\training_checkpoint_2000",
			"GAN\\DCGAN\\6_layers\\05_J\\training_checkpoint_2000",
			"GAN\\DCGAN\\6_layers\\06_S\\training_checkpoint_2000",
			"GAN\\DCGAN\\6_layers\\07_V\\training_checkpoint_2000",
			"GAN\\DCGAN\\6_layers\\08_F\\training_checkpoint_2000",
			"GAN\\DCGAN\\6_layers\\09_e\\training_checkpoint_2000",
			"GAN\\DCGAN\\6_layers\\10_j\\training_checkpoint_2000",
			"GAN\\DCGAN\\6_layers\\11_E\\training_checkpoint_2000"
		])
	'''
	test_pretrained("Classification\\WithGAN\\SplittedDataset\\ResNet34\\training_checkpoint_3000",["N","L","R","A","a","J","S","V","F","e","j","E"])
	