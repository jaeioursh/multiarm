
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv,PettingZooEnv

from pettingzoo.sisl import waterworld_v4

from rlwrapper import multiarm as sim
import numpy as np

env1=ParallelPettingZooEnv(waterworld_v4.parallel_env())
env2=sim()
for env in [env1,env2]:
	print(env.reset())
	print(env.agents)
	#print(env.observation_spaces)
	print(env.observation_space)
	print(type(env.observation_space))
	#print(env.action_spaces)
	print(env.action_space)