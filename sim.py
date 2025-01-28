import time
import math
import mujoco
import mujoco.viewer
import numpy as np

class armsim3:
	def __init__(self,view=False):
		self.m = mujoco.MjModel.from_xml_path('xarm7/scene.xml')
		self.d = mujoco.MjData(self.m)
		self.n_joints=self.m.nu
		if view:
			self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
			self.viewer.cam.distance = self.m.stat.extent * 5.0
		else:
			self.viewer = None
		self.reset()
		self.action_dim=8
		self.state_dim=39
		self.n_agents=3
		
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
			'''
			r=self.prev_dist-dists
			self.prev_dist=dists
			r*=3
			r[r<0]*=2
			'''
			r=-dists
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

class armsim2:
	def __init__(self,view=False):

		self.m = mujoco.MjModel.from_xml_path('aloha/aloha2.xml')
		self.d = mujoco.MjData(self.m)
		self.n_joints=self.m.nu
		if view:
			self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
			self.viewer.cam.distance = self.m.stat.extent * 5.0
		else:
			self.viewer = None
		self.reset()
		self.action_dim=7
		self.state_dim=15 + 2 #2 finger sensors
		self.n_agents=2
		
	def reset(self):
				#pos     quaternion
		self.time=0
		box_pos=[-1,0.5,0.23]+[0,0,0,1]
		box_vel=[0]*6
		armpos=[ 0, -0.96, 1.16, 0, -0.3, 0, 0.0084, 0.0084] #for ar m with no gripper
		self.d.qpos=np.array(box_pos+armpos*2)
		self.d.qvel=np.array(box_vel+[0]*2*len(armpos))
		self.step_start=time.time()
		self.prev_dist=None
		self.start_dist=None
		self.prev_box=None
		self.sense=None
		return self.state()
	
	def done(self):
		return self.time>1000

	def G(self):
		box_pos=self.d.qpos[:3]
		x,y,z=box_pos
		#if z<0.1:
		#	return x-0.5
		return x

	def local(self):
		box_pos=self.d.qpos[:3]
		dists=[]
		for abc in ["left/right_finger_link","right/right_finger_link"]:
			id = self.m.body(abc).id
			hand_pos=self.d.xpos[id]
			vec=hand_pos-box_pos
			dist=np.sqrt(np.sum(vec*vec))
			dists.append(dist)
		dists=np.array(dists)
		if self.prev_dist is None:
			self.prev_dist=dists
			r_pos = np.zeros(2)
		else:
			r_pos=self.prev_dist-dists

		if self.sense is not None:
			touch = self.sense.copy()/100
			#touch[touch>1]=1
			r_touch=np.array([touch[0,0]+touch[0,1],touch[1,0]+touch[1,1]])*10
		else:
			r_touch = np.zeros(2)
		if(sum(r_touch) != 0):
			print(r_touch)
		return r_touch+r_pos
	def state(self):
		self.sense=np.array(self.d.sensordata).reshape((2,2))
		pos=self.d.qpos[7:]
		vel=self.d.qvel[6:]
		box_pos=self.d.qpos[:7]
		box_vel=self.d.qvel[:6]
		box_pos=np.tile(box_pos, (2,1))
		box_vel=np.tile(box_vel, (2,1))
		pos=np.array(pos).reshape((2,8))
		vel=np.array(vel).reshape((2,8))
		state=np.concatenate([pos,vel,box_pos,box_vel],axis=1,dtype=np.float32)
		state=np.concatenate([pos,box_pos,self.sense],axis=1,dtype=np.float32)
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
	sim=armsim2(1)

	for i in range(30):
		state=sim.reset()
		done=False
		while not done:
			action=[[0]*7]*2
			state,G,reward,done=sim.step(action)
			print(state.shape)
			print(state)

	