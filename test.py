from PPO import PPO
import gymnasium as gym
import numpy as np
from PPO import PPO, Params
from multiprocessing import Pool

def test1(fname):
	env = gym.make("BipedalWalker-v3")
	env_view = gym.make("BipedalWalker-v3",render_mode="human")

	params=Params(fname)
	ppo_agent = PPO(params)
	total_steps=0
	i=0
	while total_steps<params.N_steps:
		i+=1
		for j in range(params.N_batch):
			state,info = env.reset()
			done = False
			idx=0
			while not done:
				idx+=1
				action = ppo_agent.select_action(state)
				state, reward, done, _,_ = env.step(action)
				if idx==params.max_steps:
					done=True
				ppo_agent.add_reward_terminal(reward*0.1,done)

			total_steps+=idx
		if i%1==0:
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
			#print(cumulative,total_steps,ppo_agent.action_std)
			params.writer.add_scalar("reward", cumulative, total_steps)
		#print("train")
		#print(ppo_agent.action_std)
		ppo_agent.update(total_steps)
		
	

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(test1, ["save"+str(i) for i in range(4)]))