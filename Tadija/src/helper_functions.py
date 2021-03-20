import torch
import torch.nn as nn
import simulator as sim

def initialize(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm')!=-1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)

rrpc=sim.generate_omega_function()

def ode_loss(synthetic_beats,heartbeat_type):
	delta_t=torch.tensor(1/216)

	params=sim.generate_ode_parameters(synthetic_beats.shape[0],heartbeat_type)

	x_t=torch.tensor(-0.417750770388669)
	y_t=torch.tensor(-0.9085616622823985)
	z_t=torch.tensor(-0.004551233843726818)
	t=torch.tensor(0.0)

	ode_z_signal=None
	delta_signal=None

	for i in range(215):
		delta=(synthetic_beats[:,i+1]-synthetic_beats[:,i])/delta_t
		delta=delta.view(-1,1)
		
		ode_x=sim.d_x_d_t(y_t,x_t,t,rrpc,delta_t)
		ode_y=sim.d_y_d_t(y_t,x_t,t,rrpc,delta_t)
		ode_z=sim.d_z_d_t(x_t,y_t,z_t,t,params)

		y_t=y_t+delta_t*ode_y
		x_t=x_t+delta_t*ode_x
		z_t=z_t+delta_t*ode_z
		t+=1/360

		if ode_z_signal is None:
			ode_z_signal=ode_z
			delta_signal=delta
		else:
			ode_z_signal=torch.cat((ode_z_signal,ode_z),1)
			delta_signal=torch.cat((delta_signal,delta),1)

	return delta_signal,ode_z_signal

def to_class(one_hot):
	return torch.Tensor([torch.argmax(one_hot[i]) for i in range(one_hot.size(0))]).long()
