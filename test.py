
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv,PettingZooEnv

from pettingzoo.sisl import waterworld_v4

from rlwrapper import multiarm as sim
import numpy as np

env1=PettingZooEnv(waterworld_v4.env())
env2=sim()
print(env1.reset())
print(env1.step({"pursuer_"+str(i):np.array([1]*2) for i in range(3)}))
print(env2.step({"arm_"+str(i):[1]*10 for i in range(3)}) ) 