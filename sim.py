import time
import mujoco
import mujoco.viewer
import numpy as np
from copy import deepcopy as copy

model3 = mujoco.MjModel.from_xml_path('ur5e/multiscene.xml')


class armsim:
    # view: (bool) whether or not to enable viewing
    def __init__(self, view=False):
        self.m = copy(model3)
        self.d = mujoco.MjData(self.m)
        self.n_joints = self.m.nu
        # enable viewer
        if view:
            self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
            self.viewer.cam.distance = self.m.stat.extent * 5.0
        else:
            self.viewer = None
        self.welds = [mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_EQUALITY, ABC+"_attach") for ABC in ["A", "B"]]
        self.n_agents = 2
        self.start_pos = [-0.6, 0.5, 0.15]
        self.start_quat = [0, 0, 0, 1]
        state = self.reset()
        # number of joints in an arm + 1 for gripping
        self.action_dim = self.n_joints//2+1
        self.state_dim = len(state[0])

    def reset(self):
        # reset welds to off
        for wid in self.welds:
            self.m.eq_active0[wid] = 0
            self.d.eq_active[wid] = 0
        self.time = 0
        # set positions
        box_pos = self.start_pos+self.start_quat
        box_vel = [0]*6
        armpos = [-1.5708, -1.5708, 1.5708, -1.5708, -
                  1.5708, 0]  # for ar m with no gripper
        self.d.qpos = np.array(box_pos+armpos*self.n_agents)
        self.d.qvel = np.array(box_vel+[0]*self.n_agents*len(armpos))
        self.d.ctrl[:] = 0.0
        # reset physics
        mujoco.mj_step(self.m, self.d)
        self.step_start = time.time()
        return self.state()

    def done(self):
        return self.time > 1000

    def G(self):

        box_pos = self.d.qpos[:3]
        # sets the goal position
        goal_pos = np.array(self.start_pos)
        goal_pos[2] += 1  # z coordiante
        # reward is L2 positional error
        pos_err = -.2*np.sqrt(np.sum(np.square(np.array(box_pos)-goal_pos)))

        # if no hands grasping box, set error to be large
        if sum([self.d.eq_active[wid] for wid in self.welds]):
            pos_err = -1

        return pos_err

    # local reward is distance to box
    def local(self):
        return [-dist for dist, vel in zip(self.dists, self.vels)]

    # gets state vectors (one for each agent)
    def state(self):
        self.sense = np.array(self.d.sensordata)
        box_pos = self.d.qpos[:3]
        # get end effector distances to box
        self.dists = []
        for hand in ["A", "B"]:
            hand += "_attachment_site"
            hand_pos = self.d.site(hand).xpos
            vec = hand_pos-box_pos
            dist = np.sqrt(np.sum(vec*vec))
            self.dists.append(dist)
        states = []
        self.vels = []
        dof = self.m.nu//self.n_agents
        # generate list of states
        for i in range(self.n_agents):
            # first 7 are box pos(3) and box rotation(quat 4)
            pos = self.d.qpos[7+i*dof:7+(i+1)*dof]
            # first 6 are box linear vel(3) and box rotation vel euler(3)
            vel = self.d.qpos[6+i*dof:6+(i+1)*dof]
            self.vels.append(np.mean(np.abs(vel)))
            state = np.concatenate(
                [pos, box_pos], axis=0, dtype=np.float32)
            states.append(state)
        return np.array(states)

    # step the env given action vector
    # action (2x7) matrix [-1,1] : 2 agents x (6 dof + 1 grasp action)
    # do_act: if set to false with visualization enabled, the visualization becomes interactive
    # prnt: if true, print grasping info
    def step(self, action, do_act=True, prnt=False):
        if do_act:
            action = np.array(action).reshape((self.n_agents, 7))
            grasp = action[:, -1]
            # set welds for grasping
            for g, s, wid, d in zip(grasp, self.sense, self.welds, self.dists):
                if prnt:
                    print(wid, g, s, d)
                if g > 0 and d < 0.12:  # if action is grasp and distance to box < 0.12, enable weld
                    self.d.eq_active[wid] = 1
                else:
                    self.d.eq_active[wid] = 0
            # convert action to desired angels
            action = action[:, :-1]
            action = action.flatten()*0.5+0.5
            limits = np.array(self.m.actuator_ctrlrange).T
            lower, upper = limits
            action = action*(upper-lower)+lower
            # set control angles
            self.d.ctrl = action
        # step physics 3 times
        for i in range(3):
            self.time += 1
            mujoco.mj_step(self.m, self.d)
        # run viewer if enabled
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
            time_until_next_step = self.m.opt.timestep - \
                (time.time() - self.step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step*2)
            self.step_start = time.time()
        # return state, G, local reward, done
        return self.state(), self.G(), self.local(), self.done()
