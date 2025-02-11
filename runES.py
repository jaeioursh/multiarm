from sim import armsim2,armsim3
import numpy as np
import sys
import pickle as pkl
from alignment_network import reward_align
from MAES import MAES
import multiprocessing as mp
import matplotlib.pyplot as plt 

class Params:
	def __init__(self):
		self.lr=0.0001
		self.sigma=0.005
		self.pop_size=16
		
		self.nagents=2

		self.hof_size=3
		self.hof_freq=10

		self.shape=None

def proc(args):
	env,learner,idx=args
	r=np.zeros(2)
	state=env.reset()
	done=False
	while not done:
		action=learner.act(state,idx+1)
		state,G,reward,done=env.step(action)
		r+=reward
	return r

def train(fidx):
	
	params=Params()
	env=[armsim2(0) for i in range(params.pop_size)]
	params.shape=[env[0].state_dim,64,env[0].action_dim]
	learner=MAES(params)
	print(params.lr/params.sigma)
	with mp.Pool(1) as pool:
		for gen in range(1000):
			
			args=[(env[idx],learner,idx) for idx in range(learner.popsize)]
			#R=pool.map(proc,args)
			R=[proc(arg) for arg in args]
			R=np.array(R).T
			print(gen,max(R[0]))
			learner.train(R)
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
	

if __name__ == "__main__":
	if len(sys.argv)==2:
		train(int(sys.argv[1]))
	else:    
		view(int(sys.argv[1]),int(sys.argv[2]))