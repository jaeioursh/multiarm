from PPO import PPO
import gymnasium as gym
import numpy as np
from PPO import PPO, Params
from multiprocessing import Pool
import mujoco
import mujoco.viewer
from time import sleep
def test1(fname):
	env = gym.make("BipedalWalker-v3")
	env_view = gym.make("BipedalWalker-v3",render_mode="human")

	params = Params(fname)
	params.action_std=-0.5
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
		if i%10==0:
			ppo_agent.save("./logs/"+fname+".ck")
		ppo_agent.update(total_steps)

def run_test1():		
	if 0:
		test1("save0")
	else:
		with Pool(5) as p:
			print(p.map(test1, ["save"+str(i) for i in range(4)]))

def test_mujoco():
	view=1
	m = mujoco.MjModel.from_xml_path('xarm7/scene.xml')
	d = mujoco.MjData(m)

	n_joints=m.nu
	if view:
		viewer = mujoco.viewer.launch_passive(m, d)
		viewer.cam.distance = m.stat.extent * 2.0
	else:
		viewer = None

	pos=np.array(d.qpos)
	vel=np.array(d.qvel)
	print(n_joints)
	for i in range(1000):
		d.qpos=pos
		d.qvel=vel 
		for j in range(2000000):
			act=[1]*8+[2]*8+[5]*8
			d.ctrl=act
			mujoco.mj_step(m, d)
			viewer.sync()
			sleep(1.0/30.0)



if __name__ == '__main__':
	run_test1()
