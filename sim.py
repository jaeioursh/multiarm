import time
import math
import mujoco
import mujoco.viewer
import numpy as np
from copy import deepcopy as copy

model3 = mujoco.MjModel.from_xml_path('ur5e/multiscene.xml')


class armsim3:
    def __init__(self, view=False):
        self.m = copy(model3)
        self.d = mujoco.MjData(self.m)
        self.n_joints = self.m.nu
        if view:
            self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
            self.viewer.cam.distance = self.m.stat.extent * 5.0
        else:
            self.viewer = None
        self.n_agents = 2
        self.start_pos = [-0.6, 0.5, 0.15]
        self.start_quat = [0, 0, 0, 1]
        state = self.reset()
        self.action_dim = self.n_joints//2
        self.state_dim = len(state[0])

    def reset(self):
        # pos     quaternion
        self.QD = [[], []]
        self.time = 0
        box_pos = self.start_pos+self.start_quat
        box_vel = [0]*6
        armpos = [-1.5708, -1.5708, 1.5708, -1.5708, -
                  1.5708, 0]  # for ar m with no gripper
        self.d.qpos = np.array(box_pos+armpos*self.n_agents)
        self.d.qvel = np.array(box_vel+[0]*self.n_agents*len(armpos))
        self.d.ctrl[:] = 0.0
        mujoco.mj_step(self.m, self.d)
        self.step_start = time.time()
        self.prev_dist = None
        self.prev_box = None
        self.local()
        return self.state()

    def done(self):
        return self.time > 1000

    def G(self):
        box_quat = self.d.qpos[3:7]
        box_pos = self.d.qpos[:3]
        sense = np.array(self.d.sensordata)
        goal_quat = np.array([0, 0, 0, -1])
        goal_pos = np.array(self.start_pos)
        goal_pos[2] += 1  # z coordiante
        pos_err = -np.mean(np.square(np.array(box_pos)-goal_pos))
        quat_err = -np.mean(np.square(np.array(box_quat)-goal_quat))
        dists = 0.0
        for hand in ["A_wrist_3_link", "B_wrist_3_link"]:
            hand_pos = self.d.body(hand).xpos
            vec = hand_pos-box_pos
            dist = np.sqrt(np.sum(vec*vec))
            dists += dist

        sense *= -100
        sense[sense > 1] = 1
        sense = sum(sense)
        return pos_err*10 + sense + dists*-0.05

    def local(self):
        box_pos = self.d.qpos[:3]

        return None

    def state(self):
        self.sense = np.array(self.d.sensordata)
        state = np.concatenate(
            [self.d.qpos, self.sense], axis=0, dtype=np.float32)
        return np.array([state])

    def BH(self):
        return np.array(self.d.qpos[7:])

    def step(self, action, do_act=True):
        # self.QD[0].append()
        # print(self.d.ctrl.shape)
        if do_act:
            action = np.array(action).flatten()*0.5+0.5
            limits = np.array(self.m.actuator_ctrlrange).T
            lower, upper = limits
            action = action*(upper-lower)+lower
            # action[[3,4,5,10,11,12]]=0.0 #keep wrist from rotating
            self.d.ctrl = action
        for i in range(2):
            self.time += 1
            mujoco.mj_step(self.m, self.d)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
            time_until_next_step = self.m.opt.timestep - \
                (time.time() - self.step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            self.step_start = time.time()
        return self.state(), self.G(), self.local(), self.done()


model2 = mujoco.MjModel.from_xml_path('aloha/aloha2.xml')


class armsim2:
    def __init__(self, view=False):

        self.m = copy(model2)
        self.d = mujoco.MjData(self.m)
        self.n_joints = self.m.nu
        if view:
            self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
            self.viewer.cam.distance = self.m.stat.extent * 5.0
        else:
            self.viewer = None

        self.mean = np.array(
            [9.3463612e-01, -9.1866136e-02,  4.4110820e-01,  8.4124222e-05,
             -7.6736175e-03,  5.1352440e-04,  1.7019460e-02,  1.7215589e-02,
             4.4196859e-01,  4.4599998e-01, -3.9321145e-01, -1.8936336e-04,
             1.6230130e-01, -8.9226809e-04,  4.2779222e-03,  4.3749479e-03,
             -9.3942761e-01,  5.8754832e-01,  2.2086185e-01, -2.7542081e-01,
             -9.4510145e-02, -2.0561650e-01, -2.9920354e-01, -3.4017769e-01,
             -2.1428150e-01,  6.9790728e-02]
        )
        self.std = np.array(
            [2.2264232e-01, 1.6604960e-01, 1.7565261e-01, 2.1471947e-03, 6.0605329e-02,
             7.0219748e-03, 1.2631528e-03, 1.2925949e-03, 1.7655606e+00, 1.5309929e+00,
             1.1827916e+00, 3.1204186e-02, 3.8374525e-01, 1.6116606e-01, 2.3605334e-02,
             2.2922672e-02, 3.9514229e-02, 5.9330460e-02, 7.7486612e-02, 1.8418366e-01,
             1.4370833e-01, 1.6826729e-01, 8.3697116e-01, 7.6881522e-01, 1.5808831e-01,
             1.2555440e-01]
        )
        self.reset()
        self.action_dim = self.n_joints//2
        self.state_dim = len(self.state()[0])
        self.n_agents = 2

    def reset(self):
        # pos     quaternion
        mujoco.mj_resetData(self.m, self.d)
        self.time = 0
        box_pos = [-1, 0.5, 0.13]+[0, 0, 0, 1]
        box_vel = [0]*6
        armpos = [0, -0.96, 1.16, 0, -0.3, 0, 0.0084,
                  0.0084]  # for ar m with no gripper
        self.d.qpos = np.array(box_pos+armpos*2)
        self.d.qvel = np.array(box_vel+[0]*2*len(armpos))
        self.d.ctrl[:] = 0.0
        mujoco.mj_step(self.m, self.d)
        self.step_start = time.time()
        self.prev_dist = None
        self.start_dist = None
        self.prev_box = None
        self.sense = None
        self.local()
        return self.state()

    def done(self):
        return self.time > 1000

    def G(self):
        box_pos = self.d.qpos[:3]
        x, y, z = box_pos
        if z < 0.1 or z > 1.0:
            return -100
        if self.prev_dist is not None and min(self.prev_dist) > 0.1:
            return 0.0
        return (x+1)

    def local(self):
        box_pos = self.d.qpos[:3]
        dists = []
        for left, right in [["left/left_g0", "left/right_g0"], ["right/left_g0", "right/right_g0"]]:
            lid = self.m.geom(left).id
            rid = self.m.geom(right).id
            hand_pos = (self.d.geom_xpos[lid]+self.d.geom_xpos[rid])/2
            vec = hand_pos-box_pos
            dist = np.sqrt(np.sum(vec*vec))
            dists.append(dist)
        dists = np.array(dists)
        if self.prev_dist is None:
            self.prev_dist = dists
            r_pos = np.zeros(2)
        else:
            r_pos = (self.prev_dist-dists)*20
            self.prev_dist = dists
        if self.sense is not None:
            touch = self.sense.copy()
            touch = np.abs(touch)
            touch[touch > 0.1] = 0.1
            r_touch = np.array(
                [touch[0, 0]+touch[0, 1], touch[1, 0]+touch[1, 1]])*3.0
        else:
            r_touch = np.zeros(2)

        return r_touch+r_pos

    def state(self):
        self.sense = np.array(self.d.sensordata).reshape((2, 2))*500
        dist = self.prev_dist.reshape((2, 1))
        pos = self.d.qpos[7:]
        vel = self.d.qvel[6:]
        box_pos = self.d.qpos[:7]
        box_vel = self.d.qvel[:6]
        box_pos = np.tile(box_pos, (2, 1))
        box_vel = np.tile(box_vel, (2, 1))
        pos = np.array(pos).reshape((2, 8))
        vel = np.array(vel).reshape((2, 8))
        state = np.concatenate(
            [pos, vel, box_pos, box_vel], axis=1, dtype=np.float32)
        state = np.concatenate(
            [pos, vel, box_pos, self.sense, dist], axis=1, dtype=np.float32)

        # state-=self.mean
        # state/=self.std
        # state=np.clip(state,-20,20)
        return state

    def step(self, action, do_act=True):

        # print(self.d.ctrl.shape)
        if do_act:
            action = np.array(action).flatten()*0.5+0.5
            limits = np.array(self.m.actuator_ctrlrange).T
            lower, upper = limits
            action = action*(upper-lower)+lower
            # action[[3,4,5,10,11,12]]=0.0 #keep wrist from rotating
            self.d.ctrl = action
        for i in range(3):
            self.time += 1
            mujoco.mj_step(self.m, self.d)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
            time_until_next_step = self.m.opt.timestep * \
                3 - (time.time() - self.step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            self.step_start = time.time()
        return self.state(), self.G(), self.local(), self.done()


if __name__ == "__main__":
    sim = armsim3(1)

    for i in range(30):
        state = sim.reset()
        done = False
        t = -1
        while True:
            t += 0.001
            action = [[0]*6]*2
            action[0][i] = 0
            sim.d.qpos[:7] = sim.start_pos+sim.start_quat
            state, G, reward, done = sim.step(action, False)
            print(np.round(state, 4))
