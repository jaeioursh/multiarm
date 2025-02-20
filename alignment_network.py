import numpy as np
import torch
from torch import nn
from collections import deque
from random import sample
class alignment_network(nn.Module):
	def __init__(self, params,agent_idx=0):
		super(alignment_network, self).__init__()
		state_dim = params.state_dim 

		self.use_l1 = params.use_l1
		self.use_l2 = params.use_l2
		self.l_penalty=params.l_penalty
		self.device=params.device
		hidden=params.aln_hidden
		self.batch_size=params.aln_batch
		self.train_steps=params.aln_train_steps
		self.buffer=deque(maxlen=params.deq_len)
		self.temp_buffer=[[],[],[]]
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
		#o=torch.transpose(o,1,2)
		#t=torch.transpose(t,1,2)    
		O=o-torch.transpose(o,0,1)
		T=t-torch.transpose(t,0,1)
		align = torch.mul(O,T)
		align = self.sig(align)
		loss = -torch.mean(align)#,dim=0)
		return loss

	def predict(self,r,state):
		state=torch.from_numpy(np.array(state))
		return r+self.network(state).item()
	def discount_matrix(self,N,gamma):
		mat=np.zeros((N,N))
		for i in range(N):
			mat[i,i]=1
			for j in range(1,N):
				mat[i,j]=mat[i,j-1]*gamma
		return mat
	
	def sample(self):
		RGS=None
		
		buffer=sample(self.buffer,self.batch_size)
		r=np.array([b[0] for b in buffer])
		g=np.array([b[1] for b in buffer])
		state=np.array([b[2] for b in buffer])
		r,g,state=[torch.from_numpy(_) for _ in [r,g,state]]
		return r,g,state

	def train(self):
		self.step+=1
		L=[]
		S=[]
		if len(self.buffer)<self.batch_size:
			return
		S=[]
		for i in range(self.train_steps):
			r,g,state=self.sample()
			shaping=self.network(state)
			S.append(shaping.detach().numpy())
			shaped=r+shaping
			shaped=torch.sum(shaped,dim=1)
			g=torch.sum(g,dim=1)
			self.opt.zero_grad()
			loss = self.alignment_loss(g,shaped)
			L.append(loss.detach().item())
			if self.use_l2:
				loss+=self.l_penalty*torch.mean(torch.square(shaping))
			elif self.use_l1:
				loss+=self.l_penalty*torch.mean(torch.abs(shaping))

			loss.backward()
			self.opt.step()
		return np.mean(L),np.mean(S),np.std(S),np.max(np.abs(S))


	def add_sample(self,r,g,state,done):
		self.temp_buffer[0].append([r])
		self.temp_buffer[1].append([g])
		self.temp_buffer[2].append(state)
		if done:
			self.buffer.append(self.temp_buffer)
			self.temp_buffer=[[],[],[]]


class reward_align:
	def __init__(self,params):
		self.params=params
		self.nets=[alignment_network(params,i) for i in range(params.n_agents)]

	def add(self,R,G,State,done):
		for r,state,net in zip (R,State,self.nets):
			net.add_sample(r,G,state,done)

	def shape(self,R,State):
		return [net.predict(r,state) for r,state,net in zip(R,State,self.nets)]
	
	def train(self,idx):
		agent=0
		for net in self.nets:		
			summary=net.train()
			if summary is not None:
				loss,shape_avg,shape_std,shape_max=summary
				self.params.writer.add_scalar("Shaping/Alignment_loss/"+str(agent), loss,idx)
				self.params.writer.add_scalar("Shaping/Shaping_val_max/"+str(agent), shape_max,idx)
				self.params.writer.add_scalar("Shaping/Shaping_val_avg/"+str(agent), shape_avg,idx)
				self.params.writer.add_scalar("Shaping/Shaping_val_std/"+str(agent), shape_std,idx)
			agent+=1