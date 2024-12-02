from PPO import Params
from IPPO import IPPO 
from sim import armsim


env=armsim()
params=Params(n_agents=3)
params.action_dim=10
params.state_dim=41
learner=IPPO(params)
params.N_steps=3e4

step=0
while step<params.N_steps:
    for j in range(params.N_batch):
        done=False
        state=env.reset()
        while not done:
            
            step+=1
            action=learner.act(state)
            state,reward,done=env.step(action)
            learner.add_reward_terminal([reward]*3,done)
        print(step)
    learner.train(step*4000)
    



