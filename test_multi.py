from PPO import Params
from IPPO import IPPO
from sim import armsim3 as armsim
import numpy as np
import sys
import pickle as pkl
from alignment_network import reward_align


def train():
    env = armsim(0)
    params = Params(fname="save", n_agents=env.n_agents)
    params.action_dim = env.action_dim
    params.state_dim = env.state_dim
    params.action_std = -3
    params.beta_ent = 0.0
    params.lr_actor = 0.0003
    params.N_batch = 6
    params.K_epochs = 10
    params.N_steps = 3e6

    params.write()
    learner = IPPO(params)
    step = 0
    rmax = -1e10
    data = []
    idx = 0
    while step < params.N_steps:

        for j in range(params.N_batch):
            idx += 1
            done = False
            state = env.reset()
            R = np.zeros(env.n_agents)
            cumulative_G = 0.0
            while not done:
                step += 1
                action = learner.act(state)
                state, G, reward, done = env.step(action)
                cumulative_G += G
                learner.add_reward_terminal(G, reward, done)
                R += np.array(reward)
            print(step, R, cumulative_G)
            params.writer.add_scalar("Team/Global Reward", cumulative_G, idx)
            if rmax < R[0]:
                print("Best: "+str(R[0])+"  step: "+str(j))
                learner.save("logs/a0")
                rmax = R[0]
            learner.save("logs/a1")
        # shaping.train(step)
        learner.train(step)
        if idx % 10 == 0:
            with open("logs/data.dat", "wb") as f:
                pkl.dump(data, f)


def view():
    env = armsim(1)
    params = Params(n_agents=env.n_agents)
    params.action_dim = env.action_dim
    params.state_dim = env.state_dim
    learner = IPPO(params)
    learner.load("logs/a"+sys.argv[2])
    while True:
        done = False
        state = env.reset()
        R = np.zeros(env.n_agents)
        r = []
        while not done:
            action = learner.act_deterministic(state)
            state, G, reward, done = env.step(action)
            r.append(reward[0])
            R += np.array(reward)
        print(R, max(r), min(r))


def interactive():
    env = armsim(1)
    while 1:
        env.step(None, False)
        # viewer.render()


if sys.argv[1] == "train":
    train()
elif sys.argv[1] == "view":
    view()
else:
    interactive()
