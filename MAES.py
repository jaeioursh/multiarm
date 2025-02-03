
import numpy as np
from ES import ES 

class MAES:
	def __init__(self,nagents,shape,popsize,lr,sigma):
		self.nagents=nagents
		self.popsize=popsize
		self.agents=[ES(shape,popsize,lr,sigma) for i in range(nagents)]

	def train(self,R):
		for r,agent in zip(R,self.agents):
			agent.train(r)

	def act(self,states,idx):
		action=[]
		for state,agent in zip(states,self.agents):
			if idx>=0:
				action.append( agent.pop[idx].feed(state) )
			else:
				action.append( agent.best.feed(state) )
		return np.array(action)