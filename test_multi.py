from PPO import Params
from IPPO import IPPO 
from sim import armsim2,armsim3
import numpy as np
import sys
import pickle as pkl
def train():
    env=armsim2(0)
    params=Params(fname="save",n_agents=1)#env.n_agents)
    params.action_dim=env.action_dim
    params.state_dim=env.state_dim
    params.action_std=-1.0
    params.beta_ent=0.0
    params.N_batch=2
    params.K_epochs=10
    params.N_steps=3e6
    params.write()
    learner=IPPO(params)
    step=0
    rmax=-1e10
    data=[]
    idx=0
    while step<params.N_steps:
        
        for j in range(params.N_batch):
            idx+=1
            done=False
            state=env.reset()
            R=np.zeros(env.n_agents)
            while not done:
                step+=1
                action=learner.act([state[0]])
                state,G,reward,done=env.step([action]*env.n_agents)
                data.append([state[0],G,reward[0]])
                learner.add_reward_terminal([reward[0]],done)
                R+=np.array(reward)
            print(step,R)
            
            if rmax<R[0]:
                #print("Best: "+str(rmax)+"  step: "+str(rmax))
                learner.save("logs/a0")
                rmax=R[0]
            learner.save("logs/a1")
        learner.train(step)
        if idx%10==0:
            with open("logs/data.dat","wb") as f:
                pkl.dump(data,f)
        
def view():
    env=armsim2(1)
    params=Params(n_agents=1)#env.n_agents)
    params.action_dim=env.action_dim
    params.state_dim=env.state_dim
    learner=IPPO(params)
    learner.load("logs/a"+sys.argv[1])
    while True:     
        done=False
        state=env.reset()
        R=np.zeros(env.n_agents)
        r=[]
        while not done:
            action=learner.act_deterministic([state[0]])
            state,G,reward,done=env.step([action]*env.n_agents)
            r.append(reward[0])
            R+=np.array(reward)
        print(R,max(r),min(r))


if len(sys.argv)==1:
    train()
else:    
    view()