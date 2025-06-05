from sim import armsim3 as armsim
import numpy as np
import sys
import pickle as pkl

from MAES import MHES
from MAPElites import MAPElites
from ES import MultiHeadNet, MultiHeadNetRes
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
from IK import IK


class Params:
    def __init__(self, fidx=None):
        self.lr = 0.001
        self.sigma = 0.003
        self.pop_size = 64

        self.n_agents = 2

        self.hof_size = 3
        self.hof_freq = 1000

        self.shape = None
        self.net_type = MultiHeadNetRes
        self.writer = SummaryWriter("./logs/es"+str(fidx))

    def __getstate__(self):
        return {k: v for (k, v) in self.__dict__.items() if k != "writer"}


class ParamsMAP:
    def __init__(self, fidx=None):
        self.sigma = 0.0001
        self.pop_size = 32

        self.resolution = 0.4
        self.n_agents = 2

        self.shape = None
        self.net_type = MultiHeadNet
        self.writer = SummaryWriter("./logs/map"+str(fidx))

    def __getstate__(self):
        return {k: v for (k, v) in self.__dict__.items() if k != "writer"}


def proc(args):
    env = armsim(0)
    learner, idx = args
    state = env.reset()
    done = False
    data = []
    while not done:
        action = learner.act(state, idx+1)
        state, G, reward, done = env.step(action)
        data.append([G, reward, done, state])

    return data


def proc_MAP(args):
    env = armsim(0)
    network, idx = args
    state = env.reset()
    done = False

    g = 0.0
    while not done:
        action = network.feed(state)
        state, G, reward, done = env.step(action)
        g += G

    return (G, env.BH())


def handle_data(data):

    R = []
    GG = []
    for d in data:
        r = 0.0
        states = []
        g = []
        for G, reward, done, state in d:
            g.append(G)
        GG.append(sum(g))
        R.append(r)
    R = np.array(R)
    # print(R.shape)
    G = np.array(GG)

    return G


def train(fidx):

    params = Params(fidx)

    n_agents = params.n_agents
    env = armsim(0)
    params.shape = [[env.state_dim, 64], env.n_agents, [64, env.action_dim]]
    params.state_dim = env.state_dim
    learner = MHES(params)
    print(params.lr/params.sigma)
    with mp.Pool(12) as pool:
        for gen in range(10000):

            args = [(learner, idx) for idx in range(learner.popsize)]
            # data = [proc(arg) for arg in args]
            data = pool.map(proc, args)
            G = handle_data(data)
            print(gen)
            learner.train(G)
            if gen % 10 == 0:
                with open("logs/ES"+str(fidx)+".dat", "wb") as f:
                    pkl.dump(learner, f)


def post_train(fidx):

    params = Params(fidx)
    env = armsim(0)
    print(params.lr/(params.sigma*params.pop_size))
    params.shape = [[env.state_dim, 64],
                    env.n_agents, [64, env.action_dim]]
    learner = MHES(params)
    with mp.Pool(12) as pool:
        for gen in range(10000):

            args = [(learner, idx) for idx in range(learner.popsize)]
            # data = [proc(arg) for arg in args]
            data = pool.map(proc, args)
            G = handle_data(data)
            params.writer.add_scalar("G", max(G), gen)
            print(gen)
            learner.train(G)
            if gen % 10 == 0:
                with open("logs/ES"+str(fidx)+".dat", "wb") as f:
                    pkl.dump(learner, f)


def train_map(fidx):

    params = ParamsMAP(fidx)
    n_agents = params.n_agents
    env = armsim(0)
    params.shape = [[env.state_dim, 64, 64],
                    env.n_agents, [64, env.action_dim]]
    params.state_dim = env.state_dim
    learner = MAPElites(params)

    with mp.Pool(12) as pool:
        for gen in range(10000):
            pop = [p for p in learner.policies()]
            args = [(pop[idx], idx) for idx in range(learner.popsize)]
            # data = [proc_MAP(arg) for arg in args]
            data = pool.map(proc_MAP, args)

            BH = [d[1] for d in data]
            G = [d[0] for d in data]
            improves = learner.train(BH, G)
            params.writer.add_scalar("Learner/MAP_size", len(learner.map), gen)
            params.writer.add_scalar("Learner/G", max(G), gen)
            if gen % 20 == 0:
                params.writer.add_histogram(
                    "Learner/MIN", learner.bh_data[0], gen)
                params.writer.add_histogram(
                    "Learner/MAX", learner.bh_data[1], gen)
            params.writer.add_scalar("Learner/Improvements", improves, gen)
            if gen % 10 == 0:
                with open("logs/MAP"+str(fidx)+".dat", "wb") as f:
                    pkl.dump(learner, f)


def get_data(flag=False):
    sim = armsim(flag)
    ik = IK()
    actions = []
    states = []

    targets = []
    cutoff = 200
    cutoff2 = 350
    for j in range(2):
        sim.reset()
        done = False
        i = 0

        while not done:
            i += 1
            if j == 0:
                box_quat1 = np.array([0.7071, 0, 0.7071, 0])
                box_quat2 = np.copy([0.7071, 0, -0.7071, 0])

                # sim.d.qpos[:3]  # + np.array([0.0, 0.0, 0.3])
                box_pos = sim.start_pos
                if i <= cutoff:
                    d_pos1 = -np.array([0.3, 0.0, 0.0])
                    d_pos2 = -d_pos1
                elif i <= cutoff2:
                    d_pos1 = -np.array([0.03, 0.0, 0.0])
                    d_pos2 = -d_pos1
                else:
                    d_pos1 = -np.array([0.03, 0.0, -0.2])
                    d_pos2 = np.array([0.03, 0.0, 0.2])

                ik.set_targets([box_pos+d_pos1, box_pos+d_pos2],
                               [box_quat1, box_quat2])
                action = ik.get_q(sim.m, sim.d, 2)
                if i <= cutoff*0.85:
                    action[4] = -0.6
                    action[5] = 0.75
                    action[10] = -1.5
                if i == cutoff:
                    targets.append(action)
                if i == cutoff2:
                    targets.append(action)

            else:
                if i < cutoff:
                    action = targets[0]
                elif i < cutoff2:
                    action = targets[1]
                else:
                    action = targets[2]

            if j == 1:
                actions.append(action)
                states.append(state[0])
            state, G, reward, done = sim.step(action)
            if done and j == 0:
                targets.append(action)
    states, actions = np.array(states), np.array(actions)
    return states, actions


def pretrain():
    states, actions = get_data()
    print(states.shape, actions.shape)

    params = Params(0)
    params.lr = 0.0001
    params.sigma = 0.0001
    env = armsim(0)
    params.hof_size = 3
    params.hof_freq = 50000
    print(params.lr/(params.sigma*params.pop_size))
    params.shape = [[env.state_dim, 64],
                    env.n_agents, [64, env.action_dim]]
    params.state_dim = env.state_dim
    learner = MHES(params)
    for gen in range(5000):
        learner.agent.LR *= 0.999
        learner.agent.sigma *= 0.9998
        r = []
        for idx in range(learner.popsize):
            act = learner.act(states, idx+1)
            act = act.transpose(1, 0, 2)
            act = act.reshape((act.shape[0], -1))
            diff = np.power(actions-act, 2)
            diff = -np.mean(diff)
            r.append(diff)
        print(gen, max(r), learner.agent.sigma, learner.agent.LR)
        learner.train(r)
    with open("logs/BASE.dat", "wb") as f:
        pkl.dump(learner, f)


def run():
    N = 2
    with mp.Pool(N) as pool:
        pool.map(train, range(N))


def view(fidx, idx):
    env = armsim(1)
    fname = "logs/ES"+str(fidx)+".dat"
    if fidx == 1000:
        fname = "logs/BASE.dat"
    with open(fname, "rb") as f:
        learner = pkl.load(f)
    params = learner.params
    for key, val in params.__dict__.items():
        print(key+" : "+str(val))
    for i in range(1000):
        state = env.reset()
        done = False

        info = []
        r = 0.0
        while not done:
            action = learner.act(state, idx)
            state, G, reward, done = env.step(action, prnt=True)
            r += G
            if i == 0:
                info.append(state[0])
            if not env.viewer.is_running():
                return
        print(i, r)
        if i == 0:
            info = np.array(info)

            mean = np.mean(info, axis=0)
            print(np.array2string(mean, separator=', '))
            std = np.std(info, axis=0)
            print(np.array2string(std, separator=', '))
            plt.boxplot(info)
            plt.show()


def record(fidx):
    env = armsim(0)
    with open("logs/ES"+str(fidx)+".dat", "rb") as f:
        learner = pkl.load(f)
    params = learner.params
    for key, val in params.__dict__.items():
        print(key+" : "+str(val))
    info = []
    for i in range(100):
        state = env.reset()
        done = False
        while not done:
            action = learner.act(state, 0)
            action += np.random.normal(0, float(i)/100, action.shape)
            state, G, reward, done = env.step(action)
            info.append([state, G, reward, done])
    with open("logs/train_data"+str(fidx)+".dat", "wb") as f:
        pkl.dump(info, f)


def mpproc(args):
    env = armsim(0)
    learner, idx = args
    r = np.zeros(2)
    state = env.reset()
    done = False
    gsum = 0.0
    while not done:
        action = learner.act(state, idx+1)
        state, G, reward, done = env.step(action)
        gsum += G
        r += np.array(reward)
        # r+=reward
    return r, G


if __name__ == "__main__":
    if sys.argv[1] == "pretrain":
        pretrain()
    elif sys.argv[1] == "preview":
        view(1000, 0)
    elif sys.argv[1] == "view":
        view(int(sys.argv[2]), 0)
    elif sys.argv[1] == "train":
        post_train(int(sys.argv[2]))
    elif sys.argv[1] == "data":
        get_data(1)
    else:
        print("Invalid Args")
        # record(int(sys.argv[1]))
