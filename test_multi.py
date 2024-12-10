from PPO import Params
from IPPO import IPPO 
from sim import armsim
import numpy as np
import sys
def train():
    env=armsim(0)
    params=Params(fname="save",n_agents=1)
    params.action_dim=8
    params.state_dim=39
    params.action_std=-1.0
    params.beta_ent=0.0005
    params.lr_actor=3e-4
    params.N_batch=6
    params.K_epochs=18
    params.lr_critic=params.lr_actor*10.0
    params.N_steps=1e7
    params.N_batch=5
    params.write()
    learner=IPPO(params)
    step=0
    rmax=-1e10
    while step<params.N_steps:
        for j in range(params.N_batch):
            done=False
            state=env.reset()
            R=np.zeros(3)
            while not done:
                step+=1
                action=learner.act([state[0]])
                state,G,reward,done=env.step([action]*3)
                learner.add_reward_terminal([reward[0]],done)
                R+=np.array(reward)
            print(step,R)
            
            #rmax=R[0]
            #print("Best: "+str(rmax)+"  step: "+str(rmax))
            learner.save("logs/a")
        learner.train(step)
        
def view():
    env=armsim(1)
    params=Params(n_agents=1)
    params.action_dim=8
    params.state_dim=39
    learner=IPPO(params)
    learner.load("logs/a")
    while True:     
        done=False
        state=env.reset()
        R=np.zeros(3)
        r=[]
        while not done:
            action=learner.act_deterministic([state[0]])
            state,G,reward,done=env.step([action]*3)
            r.append(reward[0])
            R+=np.array(reward)
        print(R,max(r),min(r))


if len(sys.argv)==1:
    train()
else:    
    view()