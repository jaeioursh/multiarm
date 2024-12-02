from PPO import Params
from IPPO import IPPO 
from sim import armsim


env=armsim(1)
params=Params(n_agents=3)
params.action_dim=10
params.state_dim=41
learner=IPPO(params)
params.N_steps=1e4

step=0
while step<params.N_steps:
    for j in range(params.N_batch):
        done=False
        state=env.reset()
        R=0.0
        while not done:
            step+=1
            action=learner.act(state)
            state,reward,done=env.step(action)
            learner.add_reward_terminal([reward]*3,done)
            R+=reward
        print(step,R)
    learner.train(step*4000)
    



