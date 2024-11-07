from sim import armsim as sim
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym

class multiarm(MultiAgentEnv):
	def __init__(self):
		MultiAgentEnv.__init__(self)
		self.env=sim()
		self.names=["arm_"+str(i) for i in range(3)]
		self.agents=self.names
		self.observation_spaces = {
			name: gym.spaces.Box(low=-20.0, high=20.0, shape=(41,))
			for name in self.names
		}
		self.action_spaces = {
			name: gym.spaces.Box(low=-2.0, high=2.0, shape=(10,))
			for name in self.names
		}
	def reset(self,seed=None, options=None):
		self.env.reset()
		return self.state(),self.info()
	def state(self):
		s={}
		for name,state in zip(self.names,self.env.state()):
			s[name]=state
		return s
	
	def done(self):
		done=self.env.done()
		d={"__all__":done}
		for name in self.names:
			d[name]=done
		return d
	
	def info(self):
		return {name:{} for name in self.names}	
	
	def reward(self):
		r=self.env.reward()
		return {name:r for name in self.names}	
	
	def step(self,action_dict):
		act=[action_dict[name] for name in self.names]
		self.env.step(act)
		return self.state(),self.reward(),self.done(),self.done(),self.info()
		

