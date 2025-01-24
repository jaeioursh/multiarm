import numpy as np
import torch
from torch import nn
from collections import deque
from random import sample
class alignment_network(nn.Module):
	def __init__(self, params,agent_idx=0):
		super(alignment_network, self).__init__()
		state_dim = params.state_dim 

		self.device=params.device
		hidden=params.aln_hidden
		self.train_steps=params.aln_train_steps
		self.buffer=deque(maxlen=params.deq_len)
		
		# critic
		self.network = nn.Sequential(
						nn.Linear(state_dim, hidden),
						nn.LeakyReLU(),
						nn.Linear(hidden, hidden),
						nn.LeakyReLU(),
						nn.Linear(hidden, 1)
					)
		self.opt = torch.optim.Adam(self.network.parameters(),lr=params.aln_lr)
		self.agent_idx=agent_idx
		self.step=0
		self.sig=torch.nn.Sigmoid()

	def alignment_loss(self,o, t):   
			O=o-torch.transpose(o,0,1)
			T=t-torch.transpose(t,0,1)
			align = torch.mul(O,T)
			align = self.sig(align)
			loss = -torch.mean(align)#,dim=0)
			return loss
	def predict(self,r,state):
		state=torch.from_numpy(np.array(state))
		return r+self.network(state).item()

	def sample(self):
		std_best=1e9
		RGS=None
		for i in range(10):
			buffer=sample(self.buffer,64)
			r=np.array([[b[0]] for b in buffer])
			g=np.array([[b[1]] for b in buffer])
			state=np.array([b[2] for b in buffer])
			g_std=np.std(g)
			if g_std<std_best:
				std_best=g_std
				RGS=(r,g,state)
		RGS=(torch.from_numpy(itr) for itr in RGS)
		return RGS

	def train(self):
		self.step+=1
		L=[]
		S=[]
		if len(self.buffer)<100:
			return
		for i in range(self.train_steps):
			r,g,state=self.sample()
			shaping=self.network(state)
			S.append(shaping.detach().numpy())
			self.opt.zero_grad()
			loss = self.alignment_loss(g,shaping+r)
			L.append(loss.detach().item())
			loss+=0.05*torch.mean(torch.square(shaping))
			loss.backward()
			self.opt.step()
		S=np.array(S)
		return np.mean(L),np.max(S),np.min(S)


	def add_sample(self,r,g,state):
		self.buffer.append([r,g,state])

class reward_align:
	def __init__(self,params):
		self.params=params
		self.nets=[alignment_network(params,i) for i in range(params.n_agents)]

	def add(self,R,G,State):
		for r,state,net in zip (R,State,self.nets):
			net.add_sample(r,G,state)

	def shape(self,R,State):
		return [net.predict(r,state) for r,state,net in zip(R,State,self.nets)]
	
	def train(self,idx):
		agent=0
		for net in self.nets:		
			loss,shape_max,shape_min=net.train()
			self.params.writer.add_scalar("Shaping/Alignment_loss/"+str(agent), loss,idx)
			self.params.writer.add_scalar("Shaping/Shaping_val_max/"+str(agent), shape_max,idx)
			self.params.writer.add_scalar("Shaping/Shaping_val_min/"+str(agent), shape_min,idx)
			agent+=1