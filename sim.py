import time

import mujoco
import mujoco.viewer
import numpy as np

class armsim:
	def __init__(self,view=False):
		
		self.m = mujoco.MjModel.from_xml_path('ur5e/multiscene.xml')
		self.d = mujoco.MjData(self.m)
		self.n_joints=self.m.nu
		if view:
			self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
			self.viewer.cam.distance = self.m.stat.extent * 7.0
		else:
			self.viewer = None
		self.reset()
		
	def reset(self):
		self.d.qpos=np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]*3)
		self.d.qvel=np.array([0, 0, 0, 0, 0, 0]*3)
		self.step_start=time.time()
		return self.state()

	def reward(self):
		return 0.0
	
	def state(self):
		return self.d.qpos #self.qvel

	def step(self,action):
		self.d.ctrl=np.array(action)
		mujoco.mj_step(self.m, self.d)
		if self.viewer is not None and self.viewer.is_running():
			self.viewer.sync()
			time_until_next_step = self.m.opt.timestep - (time.time() - self.step_start)
			if time_until_next_step > 0:
				time.sleep(time_until_next_step)
			self.step_start=time.time()
		return self.state(),self.reward()


if __name__ =="__main__":
	sim=armsim(1)
	state=sim.reset()
	for i in range(1000):
		action=[1]*18
		state,reward=sim.step(action)