from sim import armsim2,armsim3
import numpy as np
import sys
import pickle as pkl
from alignment_network import reward_align
from MAES import MAES
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter

class Params:
	def __init__(self,fidx=None):
		self.lr=0.0001
		self.sigma=0.005
		self.pop_size=32
		
		self.n_agents=2

		self.hof_size=3
		self.hof_freq=10

		self.shape=None

		self.aln_hidden=64
		self.aln_lr=0.0003
		self.aln_batch=16
		self.aln_train_steps=100
		self.deq_len=300
		self.use_l1=True
		self.use_l2=False
		self.l_penalty=1.0
		self.device="cpu"

		self.writer=SummaryWriter("./logs/es"+str(fidx))
	def __getstate__(self):
		return {k:v for (k, v) in self.__dict__.items() if k!="writer"}

def proc(args):
	env,learner,idx,shaping=args
	r=np.zeros(2)
	state=env.reset()
	done=False
	gsum=0.0
	while not done:
		action=learner.act(state,idx+1)
		state,G,reward,done=env.step(action)
		gsum+=G
		shaping.add(reward,G,state,done)
		shaped = shaping.shape(reward,state)
		r+=np.array(shaped)
		#r+=reward
	return r,G

def train(fidx):
	
	params=Params(fidx)
	env=armsim2(0)
	params.shape=[env.state_dim,64,env.action_dim]
	params.state_dim=env.state_dim
	learner=MAES(params)
	shaping=reward_align(params)
	print(params.lr/params.sigma)

	for gen in range(1000):
		
		args=[(env,learner,idx,shaping) for idx in range(learner.popsize)]
		RG=[proc(arg) for arg in args]
		G=[rg[1] for rg in RG]
		R=np.array([rg[0] for rg in RG]).T
		params.writer.add_scalar("Learner/G", max(G),gen)
		params.writer.add_scalar("Learner/R0", max(R[0]),gen)
		params.writer.add_scalar("Learner/R1", max(R[1]),gen)
		print(gen)
		learner.train(R)
		shaping.train(gen)
		if gen%10==0:
			with open("logs/ES"+str(fidx)+".dat","wb") as f:
				pkl.dump(learner,f)

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


if __name__ == "__main__":
	if len(sys.argv)==2:
		train(int(sys.argv[1]))
	elif len(sys.argv)>3:
		record(int(sys.argv[1]))
	else:    
		view(int(sys.argv[1]),int(sys.argv[2]))