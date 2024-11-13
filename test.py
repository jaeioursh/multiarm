from PPOold import PPO
import gymnasium as gym
import numpy as np

env = gym.make("BipedalWalker-v3")

K_epochs = 30			   # update policy for K epochs in one PPO update

eps_clip = 0.2		  # clip parameter for PPO
gamma = 0.95			# discount factor

lr_actor = 0.0003	   # learning rate for actor network
lr_critic = 0.001	   # learning rate for critic network
action_std = 0.6	  
random_seed = 0

action_dim = 4
state_dim = 24
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)
for i in range(10000):
	for j in range(5):
		state,info = env.reset()
		done = False
		cumulative=0.0
		idx=0
		while not done:
			idx+=1
			action = ppo_agent.select_action(state)
			
			state, reward, done, _,_ = env.step(action)
			cumulative+=reward
			if idx==200:
				done=True

			# saving reward and is_terminals
			ppo_agent.buffer.rewards.append(float(reward))
			ppo_agent.buffer.is_terminals.append(done)
		print(cumulative)
	print("train")
	print(ppo_agent.action_std)
	ppo_agent.update()
	ppo_agent.decay_action_std()