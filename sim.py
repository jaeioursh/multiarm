import time
import math
import mujoco
import mujoco.viewer
import numpy as np

class armsim:
	def __init__(self,view=False):
		
		#self.m = mujoco.MjModel.from_xml_path('ur5e/multiscene.xml')
		self.m = mujoco.MjModel.from_xml_path('ur5grip/scene.xml')
		self.d = mujoco.MjData(self.m)
		self.n_joints=self.m.nu
		if view:
			self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
			self.viewer.cam.distance = self.m.stat.extent * 2.0
		else:
			self.viewer = None
		self.reset()
		
	def reset(self):
				#pos     quaternion
		self.time=0
		box_pos=[-1,1,0.2]+[0,0,0,1]
		box_vel=[0]*6
		armpos=[-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0] #for ar m with no gripper
		armpos+=[0]*8 #gripper pos
		self.d.qpos=np.array(box_pos+armpos*3)
		self.d.qvel=np.array(box_vel+[0]*3*len(armpos))
		self.step_start=time.time()
		return self.state()
	
	def done(self):
		return self.time>1000

	def reward(self):
		box_pos=self.d.qpos[:3]
		x,y,z=box_pos
		if z<0.1:
			return -10
		return x
		return 0.0
	
	def state(self):
		pos=self.d.qpos[7:]
		vel=self.d.qvel[6:]
		box_pos=self.d.qpos[:7]
		box_vel=self.d.qvel[:6]
		box_pos=np.tile(box_pos, (3,1))
		box_vel=np.tile(box_vel, (3,1))
		pos=np.array(pos).reshape((3,6+8))
		vel=np.array(vel).reshape((3,6+8))
		state=np.concatenate([pos,vel,box_pos,box_vel],axis=1,dtype=np.float32)
		state=np.clip(state,-20,20)
		return  state

	def step(self,action):
		self.time+=1
		#print(self.d.ctrl.shape)
		act=[]
		for a in action:
			act.append(np.concatenate((a,a[-4:])))
		self.d.ctrl=np.array(act).flatten()
		mujoco.mj_step(self.m, self.d)
		if self.viewer is not None and self.viewer.is_running():
			self.viewer.sync()
			time_until_next_step = self.m.opt.timestep - (time.time() - self.step_start)
			if time_until_next_step > 0:
				time.sleep(time_until_next_step)
			self.step_start=time.time()
		return self.state(),self.reward(),self.done()


if __name__ =="__main__":
	sim=armsim(1)

	for i in range(30):
		state=sim.reset()
		done=False
		while not done:
			action=[[0]*10]*3
			state,reward,done=sim.step(action)
			print(state)

	