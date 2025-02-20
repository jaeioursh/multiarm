from PPO import Params
from IPPO import IPPO 
from sim import armsim2,armsim3
import numpy as np
import sys
import pickle as pkl
from alignment_network import reward_align

def train():
    env=armsim2(0)
    params=Params(fname="save",n_agents=env.n_agents)
    params.action_dim=env.action_dim
    params.state_dim=env.state_dim
    params.action_std=-0.9
    params.beta_ent=0.0
    params.N_batch=6
    params.K_epochs=10
    params.N_steps=3e6

    params.aln_hidden=64
    params.aln_lr=0.0003
    params.aln_train_steps=100
    params.deq_len=100
    params.use_l1=True
    params.use_l2=False
    params.l_penalty=0.1

    params.write()
    learner=IPPO(params)
    shaping=reward_align(params)
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
            cumulative_G=0.0
            while not done:
                step+=1
                action=learner.act(state)
                state,G,reward,done=env.step(action)
                shaping.add(reward,G,state,done)
                #reward=shaping.shape(reward,state)
                #data.append([state[0],G,reward[0]])
                learner.add_reward_terminal(reward,done)
                cumulative_G+=G
                R+=np.array(reward)
            print(step,R)
            params.writer.add_scalar("Team/Global Reward", cumulative_G ,idx)
            if rmax<R[0]:
                print("Best: "+str(R[0])+"  step: "+str(j))
                learner.save("logs/a0")
                rmax=R[0]
            learner.save("logs/a1")
        #shaping.train(step)
        learner.train(step)
        if idx%10==0:
            with open("logs/data.dat","wb") as f:
                pkl.dump(data,f)
        
def view():
    env=armsim2(1)
    params=Params(n_agents=env.n_agents)
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
            action=learner.act_deterministic(state)
            state,G,reward,done=env.step(action)
            r.append(reward[0])
            R+=np.array(reward)
        print(R,max(r),min(r))

def interactive():
    env=armsim2(1)
    while 1:
        env.step(None,False)
        #viewer.render()

if len(sys.argv)==1:
    train()
elif len(sys.argv)==3:
    interactive()
else:    
    view()