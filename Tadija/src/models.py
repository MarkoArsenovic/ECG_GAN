import torch.nn as nn
import torch

# DCGAN

class DCGenerator(nn.Module):
	def __init__(self):
		super(DCGenerator,self).__init__()
		self.main=nn.Sequential(
			nn.ConvTranspose1d(100,2048,4,1,0,bias=True),
			nn.BatchNorm1d(2048),
			nn.ReLU(True),
			nn.ConvTranspose1d(2048,1024,4,1,0,bias=True),
			nn.BatchNorm1d(1024),
			nn.ReLU(True),
			nn.ConvTranspose1d(1024,512,4,2,1,bias=True),
			nn.BatchNorm1d(512),
			nn.ReLU(True),
			nn.ConvTranspose1d(512,256,3,2,1,bias=True),
			nn.BatchNorm1d(256),
			nn.ReLU(True),
			nn.ConvTranspose1d(256,128,4,2,1,bias=True),
			nn.BatchNorm1d(128),
			nn.ReLU(True),
			nn.ConvTranspose1d(128,64,4,2,1,bias=True),
			nn.BatchNorm1d(64),
			nn.ReLU(True),
			nn.ConvTranspose1d(64,1,4,2,1,bias=True),
		)

	def forward(self,x):
		x=x.view(-1,100,1)
		x=self.main(x)
		x=x.view(-1,216)
		return x

class DCDiscriminator(nn.Module):
	def __init__(self):
		super(DCDiscriminator,self).__init__()
		self.main=nn.Sequential(
			nn.Conv1d(1,64,4,2,1,bias=False),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(64,128,4,2,1,bias=False),
			nn.BatchNorm1d(128),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(128,256,4,2,1,bias=False),
			nn.BatchNorm1d(256),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(256,512,4,2,1,bias=False),
			nn.BatchNorm1d(512),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(512,1024,4,2,1,bias=False),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(1024,1,5,2,0,bias=False),
			nn.Sigmoid(),
		)

	def forward(self,x):
		x=x.view(-1,1,216)
		return self.main(x)


# WGAN (G:7 layers, D:6 layers)

class WGenerator(nn.Module):
	def __init__(self):
		super(WGenerator,self).__init__()
		self.main=nn.Sequential(
			nn.ConvTranspose1d(100,2048,4,1,0,bias=True),
			nn.BatchNorm1d(2048),
			nn.ReLU(True),
			nn.ConvTranspose1d(2048,1024,4,1,0,bias=True),
			nn.BatchNorm1d(1024),
			nn.ReLU(True),
			nn.ConvTranspose1d(1024,512,4,2,1,bias=True),
			nn.BatchNorm1d(512),
			nn.ReLU(True),
			nn.ConvTranspose1d(512,256,3,2,1,bias=True),
			nn.BatchNorm1d(256),
			nn.ReLU(True),
			nn.ConvTranspose1d(256,128,4,2,1,bias=True),
			nn.BatchNorm1d(128),
			nn.ReLU(True),
			nn.ConvTranspose1d(128,64,4,2,1,bias=True),
			nn.BatchNorm1d(64),
			nn.ReLU(True),
			nn.ConvTranspose1d(64,1,4,2,1,bias=True),
		)

	def forward(self,x):
		x=x.view(-1,100,1)
		x=self.main(x)
		x=x.view(-1,216)
		return x

class WDiscriminator(nn.Module):
	def __init__(self):
		super(WDiscriminator,self).__init__()
		self.main=nn.Sequential(
			nn.Conv1d(1,64,4,2,1,bias=False),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(64,128,4,2,1,bias=False),
			nn.BatchNorm1d(128),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(128,256,4,2,1,bias=False),
			nn.BatchNorm1d(256),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(256,512,4,2,1,bias=False),
			nn.BatchNorm1d(512),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(512,1024,4,2,1,bias=False),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(1024,1,6,2,0,bias=False),
		)

	def forward(self,x):
		x=x.view(-1,1,216)
		x=self.main(x)
		x.view(-1,1)
		return x


# WGAN (G: 9 layers, D: 9 layers)

class W9Generator(nn.Module):
	def __init__(self):
		super(W9Generator,self).__init__()
		self.main=nn.Sequential(
			nn.ConvTranspose1d(100,2048,4,1,0,bias=True),
			nn.BatchNorm1d(2048),
			nn.ReLU(True),
			nn.ConvTranspose1d(2048,1024,4,1,0,bias=True),
			nn.BatchNorm1d(1024),
			nn.ReLU(True),
			nn.ConvTranspose1d(1024,512,4,1,0,bias=True),
			nn.BatchNorm1d(512),
			nn.ReLU(True),
			nn.ConvTranspose1d(512,256,5,1,1,bias=True),
			nn.BatchNorm1d(256),
			nn.ReLU(True),
			nn.ConvTranspose1d(256,128,4,2,1,bias=True),
			nn.BatchNorm1d(128),
			nn.ReLU(True),
			nn.ConvTranspose1d(128,64,4,1,0,bias=True),
			nn.BatchNorm1d(64),
			nn.ReLU(True),
			nn.ConvTranspose1d(64,32,4,2,1,bias=True),
			nn.BatchNorm1d(32),
			nn.ReLU(True),
			nn.ConvTranspose1d(32,16,4,2,1,bias=True),
			nn.BatchNorm1d(16),
			nn.ReLU(True),
			nn.ConvTranspose1d(16,1,4,2,1,bias=True),
		)

	def forward(self,x):
		x=x.view(-1,100,1)
		x=self.main(x)
		x=x.view(-1,216)
		return x

class W9Discriminator(nn.Module):
	def __init__(self):
		super(W9Discriminator,self).__init__()
		self.main=nn.Sequential(
			nn.Conv1d(1,16,4,2,1,bias=False),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(16,32,4,2,1,bias=False),
			nn.BatchNorm1d(32),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(32,64,4,2,1,bias=False),
			nn.BatchNorm1d(64),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(64,128,4,1,0,bias=False),
			nn.BatchNorm1d(128),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(128,256,4,2,1,bias=False),
			nn.BatchNorm1d(256),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(256,512,5,1,1,bias=False),
			nn.BatchNorm1d(512),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(512,1024,4,1,0,bias=False),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(1024,2048,4,1,0,bias=False),
			nn.ConvTranspose1d(2048,1,4,1,0,bias=False),
		)

	def forward(self,x):
		x=x.view(-1,1,216)
		x=self.main(x)
		x.view(-1,1)
		return x


# BiLSTM

class BiLSTMDCGenerator(nn.Module):
	def __init__(self):
		super(BiLSTMDCGenerator,self).__init__()

		self.lstm_cell_forward=nn.LSTMCell(5,100)
		self.lstm_cell_backward=nn.LSTMCell(5,100)

		self.lstm_cell=nn.LSTMCell(200,200)

		self.dropout=nn.Dropout(0.5)

		self.dense=nn.Linear(200,27)

		self.cnn=nn.Sequential(
			nn.ConvTranspose1d(1,128,4,2,1,bias=True),
			nn.BatchNorm1d(128),
			nn.ReLU(True),
			nn.ConvTranspose1d(128,64,4,2,1,bias=True),
			nn.BatchNorm1d(64),
			nn.ReLU(True),
			nn.ConvTranspose1d(64,1,4,2,1,bias=True),
		)

	def forward(self,x):
		hs_forward=torch.zeros(x.size(0),100)
		cs_forward=torch.zeros(x.size(0),100)
		hs_backward=torch.zeros(x.size(0),100)
		cs_backward=torch.zeros(x.size(0),100)
		hs_lstm=torch.zeros(x.size(0),200)
		cs_lstm=torch.zeros(x.size(0),200)

		torch.nn.init.kaiming_normal_(hs_forward)
		torch.nn.init.kaiming_normal_(cs_forward)
		torch.nn.init.kaiming_normal_(hs_backward)
		torch.nn.init.kaiming_normal_(cs_backward)
		torch.nn.init.kaiming_normal_(hs_lstm)
		torch.nn.init.kaiming_normal_(cs_lstm)
		
		x=x.view(20,-1,5)

		forward,backward=[],[]

		for i in range(20):
		 	hs_forward,cs_forward=self.lstm_cell_forward(x[i],(hs_forward,cs_forward))
		 	forward.append(hs_forward)
		 	
		for i in reversed(range(20)):
		 	hs_backward,cs_backward=self.lstm_cell_backward(x[i],(hs_backward,cs_backward))
		 	backward.append(hs_backward)

		backward.reverse()

		for fwd,bwd in zip(forward,backward):
			input_tensor=torch.cat((fwd,bwd),1)
			hs_lstm,cs_lstm=self.lstm_cell(input_tensor,(hs_lstm,cs_lstm))

		x=hs_lstm
		x=self.dropout(x)
		x=self.dense(x)
		x=x.view(-1,1,27)
		x=self.cnn(x)
		x=x.view(-1,216)

		return x

class BiLSTMGenerator(nn.Module):
	def __init__(self):
		super(BiLSTMGenerator,self).__init__()

		self.lstm_cell_forward=nn.LSTMCell(8,16)
		self.lstm_cell_backward=nn.LSTMCell(8,16)

		self.lstm_cell=nn.LSTMCell(32,64)

		self.dropout=nn.Dropout(0.5)

		self.dense=nn.Linear(64,1)

	def forward(self,x):
		hs_forward=torch.zeros(x.size(0),16)
		cs_forward=torch.zeros(x.size(0),16)
		hs_backward=torch.zeros(x.size(0),16)
		cs_backward=torch.zeros(x.size(0),16)
		hs_lstm=torch.zeros(x.size(0),64)
		cs_lstm=torch.zeros(x.size(0),64)

		torch.nn.init.kaiming_normal_(hs_forward)
		torch.nn.init.kaiming_normal_(cs_forward)
		torch.nn.init.kaiming_normal_(hs_backward)
		torch.nn.init.kaiming_normal_(cs_backward)
		torch.nn.init.kaiming_normal_(hs_lstm)
		torch.nn.init.kaiming_normal_(cs_lstm)
		
		x=x.view(216,-1,8)

		forward,backward=[],[]

		for i in range(216):
		 	hs_forward,cs_forward=self.lstm_cell_forward(x[i],(hs_forward,cs_forward))
		 	forward.append(hs_forward)
		 	
		for i in reversed(range(216)):
		 	hs_backward,cs_backward=self.lstm_cell_backward(x[i],(hs_backward,cs_backward))
		 	backward.append(hs_backward)

		backward.reverse()

		tmp=[]
		for fwd,bwd in zip(forward,backward):
			input_tensor=torch.cat((fwd,bwd),1)
			hs_lstm,cs_lstm=self.lstm_cell(input_tensor,(hs_lstm,cs_lstm))
			tmp.append(hs_lstm)

		x=None
		for hs in tmp:
			hs=self.dropout(hs)
			hs=self.dense(hs)
			if x==None:
				x=hs
			else:
				x=torch.cat((x,hs),1)

		return x


# Classifiers

class DCClassifier(nn.Module):
	def __init__(self,num_types):
		super(DCClassifier,self).__init__()
		self.main=nn.Sequential(
			nn.Conv1d(1,64,4,2,1,bias=False),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(64,128,4,2,1,bias=False),
			nn.BatchNorm1d(128),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(128,256,4,2,1,bias=False),
			nn.BatchNorm1d(256),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(256,512,4,2,1,bias=False),
			nn.BatchNorm1d(512),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(512,1024,4,2,1,bias=False),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Flatten(),
			nn.Linear(6144,num_types),
			nn.Sigmoid(),
		)

	def forward(self,x):
		x=x.view(-1,1,216)
		return self.main(x)


class DCClassifier2(nn.Module):
	def __init__(self,num_types):
		super(DCClassifier2,self).__init__()
		self.main=nn.Sequential(
			nn.Conv1d(1,64,4,2,1,bias=False),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(64,128,4,2,1,bias=False),
			nn.BatchNorm1d(128),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(128,256,4,2,1,bias=False),
			nn.BatchNorm1d(256),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(256,512,4,2,1,bias=False),
			nn.BatchNorm1d(512),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(512,1024,4,2,1,bias=False),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv1d(1024,64,6,2,0,bias=False),
			nn.Flatten(),
			nn.Linear(64,num_types),
			nn.Sigmoid(),
		)
	
	def forward(self,x):
		x=x.view(-1,1,216)
		return self.main(x)


# ResNet

class BasicBlock(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
		super(BasicBlock,self).__init__()

		self.conv1=nn.Sequential(
			nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding),
			nn.BatchNorm1d(out_channels),
			nn.ReLU(True),
		)

		self.conv2=nn.Sequential(
			nn.Conv1d(out_channels,out_channels,3,1,1),
			nn.BatchNorm1d(out_channels)
		)

		self.relu=nn.ReLU(True)

		if in_channels!=out_channels or (kernel_size,stride,padding)!=(3,1,1):
			self.shortcut=nn.Sequential(
				nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
				nn.BatchNorm1d(out_channels),
			)
		else:
			self.shortcut=nn.Sequential()

	def forward(self,x):
		out=self.conv1(x)
		out=self.conv2(out)
		out+=self.shortcut(x)
		out=self.relu(out)
		return out

class ResNetLayer(nn.Module):
	def __init__(self,num_blocks,in_channels,out_channels,kernel_size,stride,padding):
		super(ResNetLayer,self).__init__()
		blocks=[]
		blocks.append(BasicBlock(in_channels,out_channels,kernel_size,stride,padding))
		for i in range(num_blocks-1):
			blocks.append(BasicBlock(out_channels,out_channels,3,1,1))
		self.main=nn.Sequential(*blocks)

	def forward(self,x):
		return self.main(x)

class ResNet34(nn.Module):
	def __init__(self,num_types):
		super(ResNet34,self).__init__()
		self.main=nn.Sequential(
			nn.Conv1d(1,64,3,1,1,bias=False),
			nn.BatchNorm1d(64),
			ResNetLayer(3,64,64,4,2,1),
			ResNetLayer(4,64,128,8,4,2),
			ResNetLayer(6,128,256,8,4,2),
			ResNetLayer(3,256,512,6,1,0),
			nn.AdaptiveAvgPool1d(1),
			nn.Flatten(),
			nn.Linear(512,num_types),
			nn.Softmax(dim=1),
		)

	def forward(self,x):
		x=x.view(-1,1,216)
		return self.main(x)


# ResGAN

class BasicBlockTranspose(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
		super(BasicBlockTranspose,self).__init__()

		self.conv1=nn.Sequential(
			nn.ConvTranspose1d(in_channels,out_channels,3,1,1,bias=True),
			nn.BatchNorm1d(out_channels),
			nn.LeakyReLU(0.2,inplace=True),
		)

		self.conv2=nn.Sequential(
			nn.ConvTranspose1d(out_channels,out_channels,kernel_size,stride,padding,bias=True),
			nn.BatchNorm1d(out_channels)
		)

		self.relu=nn.LeakyReLU(0.2,inplace=True)

		if in_channels!=out_channels or (kernel_size,stride,padding)!=(3,1,1):
			self.shortcut=nn.Sequential(
				nn.ConvTranspose1d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
				nn.BatchNorm1d(out_channels),
			)
		else:
			self.shortcut=nn.Sequential()

	def forward(self,x):
		out=self.conv1(x)
		out=self.conv2(out)
		out+=self.shortcut(x)
		out=self.relu(out)
		return out

class ResNetTransposeLayer(nn.Module):
	def __init__(self,num_blocks,in_channels,out_channels,kernel_size,stride,padding):
		super(ResNetTransposeLayer,self).__init__()
		blocks=[]
		for i in range(num_blocks-1):
			blocks.append(BasicBlockTranspose(in_channels if i==0 else out_channels,out_channels,3,1,1))
		blocks.append(BasicBlockTranspose(in_channels if num_blocks==0 else out_channels,out_channels,kernel_size,stride,padding))
		self.main=nn.Sequential(*blocks)

	def forward(self,x):
		return self.main(x)

class Res34Generator(nn.Module):
	def __init__(self):
		super(Res34Generator,self).__init__()
		self.main=nn.Sequential(
			ResNetTransposeLayer(3,100,1024,6,1,0),
			ResNetTransposeLayer(6,1024,512,11,4,2),
			ResNetTransposeLayer(4,512,256,8,4,2),
			ResNetTransposeLayer(3,256,128,4,2,1),
			nn.ConvTranspose1d(128,1,3,1,1),
		)

	def forward(self,x):
		x=x.view(-1,100,1)
		return self.main(x).view(-1,216)

class Res34Discriminator(nn.Module):
	def __init__(self):
		super(Res34Discriminator,self).__init__()
		self.main=nn.Sequential(
			nn.Conv1d(1,64,3,1,1,bias=False),
			nn.BatchNorm1d(64),
			ResNetLayer(3,64,128,4,2,1),
			ResNetLayer(4,128,256,8,4,2),
			ResNetLayer(6,256,512,11,4,2),
			ResNetLayer(3,512,1024,6,1,0),
			nn.AdaptiveAvgPool1d(1),
			nn.Flatten(),
			nn.Linear(1024,1),
			nn.Sigmoid(),
		)

	def forward(self,x):
		x=x.view(-1,1,216)
		return self.main(x)
