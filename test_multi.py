from PPO import Params
from IPPO import IPPO 
from sim import armsim
import numpy as np

env=armsim(0)
params=Params(fname="save",n_agents=3)
params.action_dim=8
params.state_dim=39
learner=IPPO(params)
params.N_steps=1e7
params.N_batch=5
params.write()
step=0
while step<params.N_steps:
    for j in range(params.N_batch):
        done=False
        state=env.reset()
        R=np.zeros(3)
        while not done:
            step+=1
            action=learner.act(state)
            state,G,reward,done=env.step(action*2)
            learner.add_reward_terminal(reward,done)
            R+=np.array(reward)
        print(step,R)
    learner.train(step)
    



