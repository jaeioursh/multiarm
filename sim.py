import time
import math
import mujoco
import mujoco.viewer
import numpy as np

class armsim:
	def __init__(self,view=False):
		
		#self.m = mujoco.MjModel.from_xml_path('ur5e/multiscene.xml')
		#self.m = mujoco.MjModel.from_xml_path('ur5grip/scene.xml')
		self.m = mujoco.MjModel.from_xml_path('xarm7/scene.xml')
		self.d = mujoco.MjData(self.m)
		self.n_joints=self.m.nu
		if view:
			self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
			self.viewer.cam.distance = self.m.stat.extent * 5.0
		else:
			self.viewer = None
		self.reset()
		
	def reset(self):
				#pos     quaternion
		self.time=0
		box_pos=[-1,0.5,0.23]+[0,0,0,1]
		box_vel=[0]*6
		armpos=[0, -.247, 0, .909, 0, 1.15644, 0, 0, 0, 0, 0, 0, 0] #for ar m with no gripper
		self.d.qpos=np.array(box_pos+armpos*3)
		self.d.qvel=np.array(box_vel+[0]*3*len(armpos))
		self.step_start=time.time()
		self.prev_dist=None
		self.prev_box=None
		self.local()
		return self.state()
	
	def done(self):
		return self.time>1000

	def G(self):
		box_pos=self.d.qpos[:3]
		x,y,z=box_pos
		if z<0.1:
			return -10
		return x

	def local(self):
		box_pos=self.d.qpos[:3]
		dists=[]
		for abc in ["a","b","c"]:
			id = self.m.body('right_finger_'+abc).id
			hand_pos=self.d.xpos[id]
			vec=hand_pos-box_pos
			dist=np.sqrt(np.sum(vec*vec))
			dists.append(dist)
		dists=np.array(dists)
		if self.prev_dist is None:
			self.prev_dist=dists
			return np.zeros(3)
		else:
			r=self.prev_dist-dists
			self.prev_dist=dists
			r*=3
			r[r<0]*=2
			return r
	
	def state(self):
		pos=self.d.qpos[7:]
		vel=self.d.qvel[6:]
		box_pos=self.d.qpos[:7]
		box_vel=self.d.qvel[:6]
		box_pos=np.tile(box_pos, (3,1))
		box_vel=np.tile(box_vel, (3,1))
		pos=np.array(pos).reshape((3,13))
		vel=np.array(vel).reshape((3,13))
		state=np.concatenate([pos,vel,box_pos,box_vel],axis=1,dtype=np.float32)
		state=np.clip(state,-20,20)
		return  state

	def step(self,action):
		self.time+=1
		#print(self.d.ctrl.shape)
		self.d.ctrl=np.array(action).flatten()*3.0
		mujoco.mj_step(self.m, self.d)
		if self.viewer is not None and self.viewer.is_running():
			self.viewer.sync()
			time_until_next_step = self.m.opt.timestep - (time.time() - self.step_start)
			if time_until_next_step > 0:
				time.sleep(time_until_next_step)
			self.step_start=time.time()
		return self.state(),self.G(),self.local(),self.done()


if __name__ =="__main__":
	sim=armsim(1)

	for i in range(30):
		state=sim.reset()
		done=False
		while not done:
			action=[[0]*8]*3
			state,G,reward,done=sim.step(action)
			print(state.shape)
			print(state)

	