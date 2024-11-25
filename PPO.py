import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions.normal import Normal


################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda

device = torch.device('cpu')
if 0:
	if(torch.cuda.is_available()): 
		device = torch.device('cuda:0') 
		torch.cuda.empty_cache()
		print("Device set to : " + str(torch.cuda.get_device_name(device)))
	else:
		print("Device set to : cpu")
	print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
	def __init__(self):
		self.actions = []
		self.states = []
		self.logprobs = []
		self.rewards = []
		self.state_values = []
		self.is_terminals = []
	
	def clear(self):
		del self.actions[:]
		del self.states[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.state_values[:]
		del self.is_terminals[:]



class ActorCritic(nn.Module):
	def __init__(self, params):
		super(ActorCritic, self).__init__()
		state_dim = params.state_dim 
		action_dim = params.action_dim
		action_std_init = params.action_std
		actor_hidden = params.actor_hidden
		critic_hidden = params.critic_hidden
		active_fn = params.active_fn
		
		
		self.action_dim = action_dim
		self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
		# actor
		
		self.actor = nn.Sequential(
						nn.Linear(state_dim, actor_hidden),
						active_fn(),
						nn.Linear(actor_hidden, actor_hidden),
						active_fn(),
						nn.Linear(actor_hidden, action_dim),
						active_fn()
					)
		
		# critic
		self.critic = nn.Sequential(
						nn.Linear(state_dim, critic_hidden),
						active_fn(),
						nn.Linear(critic_hidden, critic_hidden),
						active_fn(),
						nn.Linear(critic_hidden, 1)
					)
		
	def set_action_std(self, new_action_std):
		self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

	def forward(self):
		raise NotImplementedError
	
	def act(self, state):


		action_mean = self.actor(state)
		dist = Normal(action_mean, self.action_var)
		action = dist.sample()
		action_logprob = dist.log_prob(action)
		state_val = self.critic(state)

		return action.detach(), action_logprob.detach(), state_val.detach()
	
	def evaluate(self, state, action):

		
		action_mean = self.actor(state)
		
		dist = Normal(action_mean, self.action_var)
			
		action_logprobs = dist.log_prob(action)
		dist_entropy = dist.entropy()
		state_values = self.critic(state)
		
		return action_logprobs, state_values, dist_entropy


class PPO:
	def __init__(self, params):

		self.params=params
		
		self.action_std = params.action_std

		self.gamma = params.gamma
		self.eps_clip = params.eps_clip
		self.K_epochs = params.K_epochs
		
		self.buffer = RolloutBuffer()

		self.policy = ActorCritic(params).to(device)
		self.opt_actor = torch.optim.Adam(self.policy.actor.parameters(),lr=params.lr_actor)
		self.opt_critic = torch.optim.Adam(self.policy.critic.parameters(),lr=params.lr_critic)

		self.policy_old = ActorCritic(params).to(device)
		self.policy_old.load_state_dict(self.policy.state_dict())
		
		self.MseLoss = nn.MSELoss()

	def set_action_std(self, new_action_std):
		self.action_std = new_action_std
		self.policy.set_action_std(new_action_std)
		self.policy_old.set_action_std(new_action_std)
	
	

	def decay_action_std(self):
		self.set_action_std(max(self.action_std-self.params.decay_rate,0.1))
		
	def select_action(self, state):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(device)
			action, action_logprob, state_val = self.policy_old.act(state)

		self.buffer.states.append(state)
		self.buffer.actions.append(action)
		self.buffer.logprobs.append(action_logprob)
		self.buffer.state_values.append(state_val)

		return action.detach().cpu().numpy().flatten()

	def deterministic_action(self, state):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(device)
			action = self.policy_old.actor(state)

		return action.detach().cpu().numpy().flatten()
	
	def add_reward_terminal(self,reward,done):
		self.buffer.rewards.append(float(reward))
		self.buffer.is_terminals.append(done)

	def update(self):
		# Monte Carlo estimate of returns
		rewards = []
		discounted_reward = 0
		for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.gamma * discounted_reward)
			rewards.insert(0, discounted_reward)
			
		# Normalizing the rewards
		rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		# convert list to tensor
		old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
		old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
		old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
		old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

		# calculate advantages
		advantages = rewards.detach() - old_state_values.detach()
		advantages=advantages.reshape((-1,1))
		#advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
		# Optimize policy for K epochs
		for _ in range(self.K_epochs):

			# Evaluating old actions and values
			logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

			# match state_values tensor dimensions with rewards tensor
			state_values = torch.squeeze(state_values)
			
			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs.detach())

			# Finding Surrogate Loss  
			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

			# final loss of clipped objective PPO
			loss_actor = -torch.min(surr1, surr2)  - self.params.beta_ent * dist_entropy

			loss_critic= 0.5 * self.MseLoss(state_values, rewards)
			
			# take gradient step
			self.opt_actor.zero_grad()
			loss_actor.mean().backward()
			self.opt_actor.step()

			self.opt_critic.zero_grad()
			loss_critic.backward()
			self.opt_critic.step()
			
		# Copy new weights into old policy
		self.policy_old.load_state_dict(self.policy.state_dict())

		# clear buffer
		self.buffer.clear()
	
	def save(self, checkpoint_path):
		torch.save(self.policy_old.state_dict(), checkpoint_path)
   
	def load(self, checkpoint_path):
		self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
		self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

class Params:
	def __init__(self):	
		self.K_epochs = 10			   # update policy for K epochs in one PPO update
		self.N_batch=4
		self.N_steps=1e6
		self.eps_clip = 0.2		  # clip parameter for PPO
		self.gamma = 0.99			# discount factor

		self.lr_actor = 0.0003	   # learning rate for actor network
		self.lr_critic = 0.001	   # learning rate for critic network
		self.action_std = 0.6	  
		self.decay_rate=0.0005
		self.random_seed = 0


		self.action_dim = 4
		self.state_dim = 24

		self.actor_hidden = 64
		self.critic_hidden = 64
		#self.active_fn = nn.LeakyReLU
		self.active_fn = nn.Tanh
		self.beta_ent=0.01


if __name__ =="__main__":
	import gymnasium as gym
	import numpy as np
	env = gym.make("BipedalWalker-v3")
	env_view = gym.make("BipedalWalker-v3",render_mode="human")

	params=Params()
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
				ppo_agent.add_reward_terminal(reward,done)
				if idx==600:
					done=True
			total_steps+=idx
		if i%3==0:
			state,info = env.reset()
			done = False
			idx=0
			cumulative=0
			while not done:
				idx+=1
				action = ppo_agent.deterministic_action(state)
				state, reward, done, _,_ = env.step(action)
				cumulative+=reward
				if idx==600:
					done=True
			print(cumulative,total_steps,ppo_agent.action_std)
		#print("train")
		#print(ppo_agent.action_std)
		ppo_agent.update()
		ppo_agent.decay_action_std()
	