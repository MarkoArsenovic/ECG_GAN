from models import ResNet34
from build_dataset import build_mixed_dataset
from helper_functions import initialize,to_class
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import os
import matplotlib.pyplot as plt


def train_resnet34_classification_no_gan(batch_size,train_steps,model_dir,heartbeat_types):
	num_b=len(heartbeat_types)
	classifier=ResNet34(num_b)
	classifier.apply(initialize)

	#cross_entropy_loss=nn.BCELoss()
	cross_entropy_loss=nn.CrossEntropyLoss()
	optimizer=optim.SGD(classifier.parameters(),lr=0.0002)

	train,validation,test=build_mixed_dataset(heartbeat_types)
	train_size=len(train[0])
	
	train=(torch.Tensor(train[0]),torch.Tensor(train[1]))
	validation=(torch.Tensor(validation[0]),torch.Tensor(validation[1]))
	test=(torch.Tensor(test[0]),torch.Tensor(test[1]))

	ptr=0

	loss_hist=[]
	accuracy_hist=[]
	validation_hist=[]
	
	steps=0
	while True:
		if steps>=train_steps:
			break
		steps+=1

		heartbeats,one_hot=train[0][ptr:min(ptr+batch_size,train_size)],train[1][ptr:min(ptr+batch_size,train_size)]
		ptr+=batch_size
		if ptr>=train_size:
			ptr=0
		
		classifier.zero_grad()

		predictions=classifier(heartbeats)
		loss=cross_entropy_loss(predictions,to_class(one_hot))

		loss.backward()
		optimizer.step()

		with torch.no_grad():
			predictions=classifier(heartbeats)

		correct=0
		for i in range(len(one_hot)):
			k,l=0,0
			for j in range(num_b):
				if predictions[i][j]>predictions[i][k]:
					k=j
				if one_hot[i][j]>one_hot[i][k]:
					l=j
			if k==l:
				correct+=1

		accuracy=correct/len(one_hot)
		accuracy_hist.append(accuracy)
		
		loss_hist.append(loss.item())

		if steps%15==1:
			with torch.no_grad():
				val_pred=classifier(validation[0])
				correct=0
				for i in range(len(validation[0])):
					k,l=0,0
					for j in range(num_b):
						if val_pred[i][j]>val_pred[i][k]:
							k=j
						if validation[0][i][j]>validation[1][i][l]:
							l=j
					if k==l:
						correct+=1
				val=correct/len(validation[0])
				validation_hist.append(val)
			print("Step {} done. Loss={:.4f}, accuracy={:.4f}, validation={:.4f}".format(steps,loss.item(),accuracy,val))
		
		else:
			if validation_hist:
				validation_hist.append(validation_hist[-1])
			else:
				validation_hist.append(0)
			print("Step {} done. Loss={:.4f}, accuracy={:.4f}".format(steps,loss.item(),accuracy))
		
		if steps%100==0:
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

		if steps%1000==0:
			torch.save({
					'classifier_state_dict': classifier.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
				},os.path.join(model_dir,'training_checkpoint_{}'.format(steps)))

	torch.save({
			'classifier_state_dict': classifier.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
		},os.path.join(model_dir,'trained_classifier'))
	
	test_x=test[0]
	test_y=test[1]

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
			print("{:8} ".format(confusion_matrix[i][j]),end=" ")
		print()


if __name__=="__main__":
	train_resnet34_classification_no_gan(200,100,'Classification\\WithoutGAN\\MixedDataset\\ResNet34_try03',["N","L","R","A","a","J","S","V","F","e","j","E"])
