import torch
import numpy as np
import math
import matplotlib.pyplot as plt

def rrprocess(n,f1,f2,c1,c2,lfhfratio,hrmean,hrstd,sf):
	sig2=1.0
	sig1=lfhfratio
	
	df=sf/float(n)
	
	f=np.linspace(0,0.5,n)
	amplitudes=np.linspace(0,1,n)
	phases=np.linspace(0,1,n)*2*np.pi
	
	complex_series=[complex(amplitudes[i]*np.cos(phases[i]),amplitudes[i]*np.sin(phases[i])) for i in range(len(phases))]
    
	T=np.fft.ifft(complex_series,n)
	T=T.real

	rrmean=60.0/hrmean
	rrstd=60.0*hrstd/(hrmean*hrmean)

	std=np.std(T)
	ratio=rrstd/std
	T=ratio*T
	T+=rrmean
	return T

def generate_omega_function():
	return torch.Tensor(rrprocess(216,0.1,0.25,0.01,0.01,0.5,60,1,512))

def mit_bih_to_aami(mit_bih):
	if mit_bih in "NLRje":
		return "N"
	elif mit_bih in "aSAJ":
		return "S"
	elif mit_bih in "EV":
		return "V"
	elif mit_bih in "F":
		return "F"
	else:
		return "Q"

typical_ode_params={
	'N':[0.7,0.25,-0.5*math.pi,-7.0,0.1,-15.0*math.pi/180.0,30.0,0.1,0.0,-3.0,0.1,15.0*math.pi/180.0,0.2,0.4,160.0*math.pi/180.0],
	'S':[0.2,0.25,-0.5*math.pi,-1.0,0.1,-15.0*math.pi/180.0,30.0,0.1,0.0,-10.0,0.1,15.0*math.pi/180.0,0.2,0.4,160.0*math.pi/180.0],
	'V':[0.1,0.6,-0.5*math.pi,0.0,0.1,-15.0*math.pi/180.0,30.0,0.1,0.0,-10.0,0.1,15.0*math.pi/180.0,0.5,0.2,160.0*math.pi/180.0],
	'F':[0.8,0.25,-0.5*math.pi,-10.0,0.1,-15.0*math.pi/180.0,30.0,0.1,0.03*math.pi/180.0,-10.0,0.1,15.0*math.pi/180.0,0.5,0.2,160.0*math.pi/180.0],
}

def typical_params(heartbeat_type):
	aami=mit_bih_to_aami(heartbeat_type)
	assert aami in typical_ode_params
	return torch.Tensor(typical_ode_params[aami])

def generate_ode_parameters(batch_size,heartbeat_type):
	noise=torch.Tensor(np.random.normal(0,0.1,(batch_size,15)))
	return noise*0.1+typical_params(heartbeat_type)

def d_x_d_t(y,x,t,rrpc,delta_t):
	alpha=1-((x*x)+(y*y))**0.5

	cast=(t/delta_t).type(torch.IntTensor)
	tensor_temp=1+cast
	tensor_temp=tensor_temp%len(rrpc)
	if rrpc[tensor_temp]==0:
		omega=(2.0*math.pi/1e-3)
	else:
		omega=(2.0*math.pi/rrpc[tensor_temp])

	f_x=alpha*x-omega*y
	return f_x

def d_y_d_t(y,x,t,rrpc,delta_t):
	alpha=1-((x*x)+(y*y))**0.5

	cast=(t/delta_t).type(torch.IntTensor)
	tensor_temp=1+cast
	tensor_temp=tensor_temp%len(rrpc)
	if rrpc[tensor_temp]==0:
		omega=(2.0*math.pi/1e-3)
	else:
		omega=(2.0*math.pi/rrpc[tensor_temp])

	f_y=alpha*y+omega*x
	return f_y

def d_z_d_t(x,y,z,t,params):
	A=0.005
	f2=0.25

	a_p,a_q,a_r,a_s,a_t=params[:,0],params[:,3],params[:,6],params[:,9],params[:,12]
	b_p,b_q,b_r,b_s,b_t=params[:,1],params[:,4],params[:,7],params[:,10],params[:,13]

	theta_p,theta_q,theta_r,theta_s,theta_t=params[:,2],params[:,5],params[:,8],params[:,11],params[:,14]

	a_p=a_p.view(-1,1)
	a_q=a_q.view(-1,1)
	a_r=a_r.view(-1,1)
	a_s=a_s.view(-1,1)
	a_t=a_t.view(-1,1)

	b_p=b_p.view(-1,1)
	b_q=b_q.view(-1,1)
	b_r=b_r.view(-1,1)
	b_s=b_s.view(-1,1)
	b_t=b_t.view(-1,1)

	theta_p=theta_p.view(-1,1)
	theta_q=theta_q.view(-1,1)
	theta_r=theta_r.view(-1,1)
	theta_s=theta_s.view(-1,1)
	theta_t=theta_t.view(-1,1)

	theta=torch.atan2(y,x)
	delta_theta_p=torch.fmod(theta-theta_p,2*math.pi)
	delta_theta_q=torch.fmod(theta-theta_q,2*math.pi)
	delta_theta_r=torch.fmod(theta-theta_r,2*math.pi)
	delta_theta_s=torch.fmod(theta-theta_s,2*math.pi)
	delta_theta_t=torch.fmod(theta-theta_t,2*math.pi)

	z_p=a_p*delta_theta_p*torch.exp((-delta_theta_p*delta_theta_p/(2*b_p*b_p)))
	z_q=a_q*delta_theta_q*torch.exp((-delta_theta_q*delta_theta_q/(2*b_q*b_q)))
	z_r=a_r*delta_theta_r*torch.exp((-delta_theta_r*delta_theta_r/(2*b_r*b_r)))
	z_s=a_s*delta_theta_s*torch.exp((-delta_theta_s*delta_theta_s/(2*b_s*b_s)))
	z_t=a_t*delta_theta_t*torch.exp((-delta_theta_t*delta_theta_t/(2*b_t*b_t)))

	z_0_t=(A*torch.sin(2*math.pi*f2*t))

	f_z=-1*(z_p+z_q+z_r+z_s+z_t)-(z-z_0_t)
	return f_z

rrpc=generate_omega_function()

def ode_loss(synthetic_beats,heartbeat_type):
	delta_t=torch.tensor(1/216)

	params=generate_ode_parameters(synthetic_beats.shape[0],heartbeat_type)

	x_t=torch.tensor(-0.417750770388669)
	y_t=torch.tensor(-0.9085616622823985)
	z_t=torch.tensor(-0.004551233843726818)
	t=torch.tensor(0.0)

	ode_z_signal=None
	delta_signal=None

	for i in range(215):
		delta=(synthetic_beats[:,i+1]-synthetic_beats[:,i])/delta_t
		delta=delta.view(-1,1)
		#z_t=synthetic_beats[:,i].view(-1,1)

		ode_x=d_x_d_t(y_t,x_t,t,rrpc,delta_t)
		ode_y=d_y_d_t(y_t,x_t,t,rrpc,delta_t)
		ode_z=d_z_d_t(x_t,y_t,z_t,t,params)

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

if __name__=="__main__":
	delta_signal,ode_z_signal=ode_loss(torch.Tensor([[0.0 for _ in range(216)]]),"N")
	plt.plot(ode_z_signal[0])
	plt.show()