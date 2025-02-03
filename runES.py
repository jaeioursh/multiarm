from sim import armsim2,armsim3
import numpy as np
import sys
import pickle as pkl
from alignment_network import reward_align
from MAES import MAES

def train():
	env=armsim2(0)
	shape=[env.state_dim,64,env.action_dim]
	learner=MAES(2,shape,popsize=16,lr=0.0001,sigma=0.005)
	for gen in range(1000):
		R=[]
		for idx in range(learner.popsize):
			r=np.zeros(2)
			state=env.reset()
			done=False
			while not done:
				action=learner.act(state,idx)
				state,G,reward,done=env.step(action)
				r+=reward
			R.append(r)
		R=np.array(R).T
		print(gen,max(R[0]))
		learner.train(R)
		if gen%20==0:
			with open("logs/ES.dat","wb") as f:
				pkl.dump(learner,f)
	
def view(idx):
	env=armsim2(1)
	with open("logs/ES.dat","rb") as f:
		learner=pkl.load(f)
	for idx in range(1000):
		state=env.reset()
		done=False
		r=np.zeros(2)
		while not done:
			action=learner.act(state,idx)
			state,G,reward,done=env.step(action)
			r+=reward
		print(r)
	

if __name__ == "__main__":
	if len(sys.argv)==1:
		train()
	else:    
		view(int(sys.argv[1]))