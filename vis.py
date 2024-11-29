from PPO import PPO
import gymnasium as gym
import numpy as np
from PPO import PPO, Params
from multiprocessing import Pool


env = gym.make("BipedalWalker-v3",render_mode="human")

params=Params("test")
ppo_agent = PPO(params)
fname="save0"
ppo_agent.load("./logs/"+fname+".ck")
total_steps=0
i=0
while total_steps<params.N_steps:
    state,info = env.reset()
    done = False
    idx=0
    cumulative=0
    while not done:
        idx+=1
        action = ppo_agent.deterministic_action(state)
        state, reward, done, _,_ = env.step(action)
        cumulative+=reward
        if idx==params.max_steps:
            done=True


    