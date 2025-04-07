from PPO import PPO
import gymnasium as gym
import numpy as np
from PPO import PPO, Params
from multiprocessing import Pool
import mujoco
import mujoco.viewer
from time import sleep
from sim import armsim3 as armsim
from IK import IK
from runES import get_data
import torch
import torch.nn as nn


class Net():
    def __init__(self):
        self.device = "cpu"
        state_dim = 21
        action_dim = 12
        LR = 1e-3
        active_fn = nn.LeakyReLU
        hidden = 64
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            active_fn(),
            nn.Linear(hidden, hidden),
            active_fn(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )
        self.loss = nn.MSELoss()
        self.opt = torch.optim.Adam(
            self.actor.parameters(), lr=LR)

    def train(self, X, Y):
        y = self.actor(X)
        loss = self.loss(y, Y)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return loss.item()


def test1(fname):
    env = gym.make("Hopper-v5")
    env_view = gym.make("Hopper-v5", render_mode="human")

    params = Params(fname)
    params.action_std = -0.2
    params.action_dim = 3
    params.state_dim = 11
    params.K_epochs = 10
    params.beta_ent = 0.0
    ppo_agent = PPO(params)
    total_steps = 0
    i = 0
    while total_steps < params.N_steps:
        i += 1
        for j in range(params.N_batch):
            state, info = env.reset()
            done = False
            idx = 0
            while not done:
                idx += 1
                action = ppo_agent.select_action(state)
                state, reward, done, _, _ = env.step(action)
                if idx == params.max_steps:
                    done = True
                ppo_agent.add_reward_terminal(reward, done)

            total_steps += idx
        if i % 1 == 0:
            state, info = env.reset()
            done = False
            idx = 0
            cumulative = 0
            while not done:
                idx += 1
                action = ppo_agent.deterministic_action(state)
                state, reward, done, _, _ = env.step(action)
                cumulative += reward
                if idx == params.max_steps:
                    done = True
            # print(cumulative,total_steps,ppo_agent.action_std)
            params.writer.add_scalar("reward", cumulative, total_steps)
        # print("train")
        # print(ppo_agent.action_std)
        if i % 10 == 0:
            ppo_agent.save("./logs/"+fname+".ck")
        ppo_agent.update(total_steps)


def run_test1():
    if 0:
        test1("save0")
    else:
        with Pool(5) as p:
            print(p.map(test1, ["save"+str(i) for i in range(4)]))


def test_mujoco():
    view = 1
    m = mujoco.MjModel.from_xml_path('xarm7/scene.xml')
    d = mujoco.MjData(m)

    n_joints = m.nu
    if view:
        viewer = mujoco.viewer.launch_passive(m, d)
        viewer.cam.distance = m.stat.extent * 2.0
    else:
        viewer = None

    pos = np.array(d.qpos)
    vel = np.array(d.qvel)
    print(n_joints)
    for i in range(1000):
        d.qpos = pos
        d.qvel = vel
        for j in range(2000000):
            act = [1]*8+[2]*8+[5]*8
            d.ctrl = act
            mujoco.mj_step(m, d)
            viewer.sync()
            sleep(1.0/30.0)


def test_arch():
    states, actions = get_data()
    states, actions = torch.tensor(states, dtype=torch.float32), torch.tensor(
        actions, dtype=torch.float32)
    net = Net()
    for i in range(10000):
        print(net.train(states, actions))


def IK_test():

    sim = armsim(True)
    ik = IK()
    targets = []
    cutoff = 200
    for j in range(10000):
        sim.reset()
        done = False
        i = 0

        while not done:
            i += 1
            if j == 0:
                box_quat1 = np.array([0.7071, 0, 0.7071, 0])
                box_quat2 = np.copy([0.7071, 0, -0.7071, 0])

                box_pos = sim.d.qpos[:3]  # + np.array([0.0, 0.0, 0.3])
                if i < cutoff:
                    d_pos = np.array([0.3, 0.0, 0.0])
                else:
                    d_pos = np.array([0.1, 0.0, 0.0])
                ik.set_targets([box_pos-d_pos, box_pos+d_pos],
                               [box_quat1, box_quat2])
                action = ik.get_q(sim.m, sim.d, 2)
                if i == cutoff:
                    targets.append(action)
                if i < cutoff:
                    action[4] = -0.6
                    action[5] = 0.75
                    action[10] = -1.5
            else:
                if i < cutoff:
                    action = targets[0]
                else:
                    action = targets[1]

            state, G, reward, done = sim.step(action)
            if done and j == 0:
                targets.append(action)

        print(action[:6])


if __name__ == '__main__':
    # IK_test()
    test_arch()
