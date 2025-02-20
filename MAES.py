
import numpy as np
from ES import ES 

class MAES:
	def __init__(self,params):
		self.params=params
		self.n_agents=params.n_agents
		self.popsize=params.pop_size
		self.agents=[ES(params) for i in range(params.n_agents)]

	def train(self,R):
		for r,agent in zip(R,self.agents):
			agent.train(r)

	def act(self,states,idx):
		action=[]
		for state,agent in zip(states,self.agents):
			if idx>0:
				action.append( agent.pop[idx-1].feed(state) )
			elif idx==0:
				action.append( agent.best.feed(state) )
			else:
				action.append( agent.hof[abs(idx)-1].feed(state) )
		return np.array(action)