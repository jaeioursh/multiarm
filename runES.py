from sim import armsim2,armsim3
import numpy as np
import sys
import pickle as pkl
from alignment_network import reward_align
from MAES import MAES
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp

class Params:
	def __init__(self,fidx=None):
		self.lr=0.0001
		self.sigma=0.005
		self.pop_size=32
		
		self.n_agents=2

		self.hof_size=3
		self.hof_freq=30

		self.shape=None

		self.aln_hidden=64
		self.aln_lr=0.00001
		self.aln_batch=16
		self.aln_train_steps=2
		self.deq_len=32
		self.use_l1=False
		self.use_l2=False
		self.use_weighting=True
		self.l_penalty=10.0
		self.update_frq=10
		self.device="cpu"

		self.writer=SummaryWriter("./logs/es"+str(fidx))
	def __getstate__(self):
		return {k:v for (k, v) in self.__dict__.items() if k!="writer"}

def proc(args):
	env=armsim2(0)
	learner,idx=args
	state=env.reset()
	done=False
	data=[]
	while not done:
		action=learner.act(state,idx+1)
		state,G,reward,done=env.step(action)
		data.append([state,G,reward,done])

	return data

def handle_data(data,shaping,params,gen):
	R=[]
	Gs=[]
	GG=[]
	for d in data:
		r=[]
		states=[]
		g=[]
		for state,G,reward,done in d:
			shaping.add(reward,G,state,done)
			states.append(state)
			g.append(G)
			r.append(reward)
		GG.append(sum(g))
		states,g,r=[np.array(itr) for itr in (states,g,r)]
		shaped=shaping.shape([states[:,0,:],states[:,1,:]])
		if params.use_weighting:
			shaped=np.array(shaped).T[0]
			g=np.array([g,g]).T
			rshape=r*shaped+g*(1-shaped)
			rshape=r+g
			r=np.sum(rshape,axis=0)
		else:
			shaped=np.sum(shaped,axis=1).flatten()
			r=np.sum(r,axis=0)+shaped
		R.append(r)
	R=np.array(R).T
	#print(R.shape)
	params.writer.add_scalar("Learner/G", max(GG),gen)
	params.writer.add_scalar("Learner/R0", max(R[0]),gen)
	params.writer.add_scalar("Learner/R1", max(R[1]),gen)
	G=np.array(GG)
	return R,G

def train(fidx):
	
	params=Params(fidx)
	env=armsim2(0)
	params.shape=[env.state_dim,64,env.action_dim]
	params.state_dim=env.state_dim
	learner=MAES(params)
	shaping=reward_align(params)
	print(params.lr/params.sigma)
	with mp.Pool(12) as pool:
		for gen in range(10000):
			
			args=[(learner,idx) for idx in range(learner.popsize)]
			#data=[proc(arg) for arg in args]
			data=pool.map(proc,args)
			R,G=handle_data(data,shaping,params,gen)
			print(gen)
			learner.train(R,G)
			if gen%params.update_frq==0:
				shaping.train(gen)
			if gen%10==0:
				with open("logs/ES"+str(fidx)+".dat","wb") as f:
					pkl.dump(learner,f)
def run():
	N=2
	with mp.Pool(N) as pool:
		pool.map(train,range(N))

def view(fidx,idx):
	env=armsim2(1)
	with open("logs/ES"+str(fidx)+".dat","rb") as f:
		learner=pkl.load(f)
	params=learner.params
	for key,val in params.__dict__.items():
		print(key+" : "+str(val))
	for i in range(1000):
		state=env.reset()
		done=False
		r=np.zeros(2)
		info=[]
		while not done:
			action=learner.act(state,idx)
			state,G,reward,done=env.step(action)
			if i==0:
				info.append(state[0])
			r+=reward
			if not env.viewer.is_running():
				return
		if i==0:
			info=np.array(info)
			
			mean=np.mean(info,axis=0)
			print(np.array2string(mean, separator=', '))
			std=np.std(info,axis=0)
			print(np.array2string(std, separator=', '))
			plt.boxplot(info)
			plt.show()
		print(r)

def record(fidx):
	env=armsim2(0)
	with open("logs/ES"+str(fidx)+".dat","rb") as f:
		learner=pkl.load(f)
	params=learner.params
	for key,val in params.__dict__.items():
		print(key+" : "+str(val))
	info=[]
	for i in range(100):
		state=env.reset()
		done=False
		while not done:
			action=learner.act(state,0)
			action+=np.random.normal(0,float(i)/100,action.shape)
			state,G,reward,done=env.step(action)
			info.append([state,G,reward,done])
	with open("logs/train_data"+str(fidx)+".dat","wb") as f:
		pkl.dump(info,f)


def mpproc(args):
	env=armsim2(0)
	learner,idx=args
	r=np.zeros(2)
	state=env.reset()
	done=False
	gsum=0.0
	while not done:
		action=learner.act(state,idx+1)
		state,G,reward,done=env.step(action)
		gsum+=G
		r+=np.array(reward)
		#r+=reward
	return r,G

def mptest():
	
	params=Params(-1)
	env=armsim2(0)
	params.shape=[env.state_dim,64,env.action_dim]
	params.state_dim=env.state_dim
	params.pop_size=16
	learner=MAES(params)
	shaping=reward_align(params)
	print(params.lr/params.sigma)
	with mp.Pool(4) as pool:
		for gen in range(4):		
			args=[(learner,idx) for idx in range(learner.popsize)]
			RG=pool.map(mpproc,args)
			R=np.array([rg[0] for rg in RG]).T
			print(gen)
			learner.train(R)


if __name__ == "__main__":

	if len(sys.argv)==2:
		train(int(sys.argv[1]))
	elif len(sys.argv)>3:
		record(int(sys.argv[1]))
	else:    
		view(int(sys.argv[1]),int(sys.argv[2]))
	