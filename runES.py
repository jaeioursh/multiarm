from sim import armsim2,armsim3
import numpy as np
import sys
import pickle as pkl
from alignment_network import reward_align
from MAES import MAES
class Params:
	def __init__(self):
		self.lr=0.0001
		self.sigma=0.005
		self.pop_size=64
		
		self.nagents=2

		self.hof_size=3
		self.hof_freq=10

		self.shape=None



def train(fidx):
	env=armsim2(0)
	params=Params()
	params.shape=[env.state_dim,64,env.action_dim]
	learner=MAES(params)
	print(params.lr/params.sigma)
	for gen in range(1000):
		R=[]
		for idx in range(learner.popsize):
			r=np.zeros(2)
			state=env.reset()
			done=False
			while not done:
				action=learner.act(state,idx+1)
				state,G,reward,done=env.step(action)
				r+=reward
			R.append(r)
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
		while not done:
			action=learner.act(state,idx)
			state,G,reward,done=env.step(action)
			r+=reward
			if not env.viewer.is_running():
				return
		print(r)
	

if __name__ == "__main__":
	if len(sys.argv)==2:
		train(int(sys.argv[1]))
	else:    
		view(int(sys.argv[1]),int(sys.argv[2]))