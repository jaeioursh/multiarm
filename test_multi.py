from PPO import Params
from IPPO import IPPO 
from sim import armsim


env=armsim()
params=Params(n_agents=3)
params.action_dim=10
params.state_dim=41
learner=IPPO(params)


for i in range(int(params.N_steps)):
    state=env.reset()
    done=False
    step=0
    for j in range(params.N_batch):
        while not done:
            step+=1
            action=learner.act(state)
            state,reward,done=env.step(action)
            learner.add_reward_terminal([reward]*3,done)
    learner.train(i*4000)
    



