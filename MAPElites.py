from ES import MultiHeadNet
import random
import numpy as np


class MAPElites:
    def __init__(self, params):
        """
        initializes map elites, w map
        args:
            params: class of params to use for map elites
        returns:
            none
        """
        self.net_type = params.net_type
        self.map = {}
        self.parent = self.net_type(params.shape)

        self.popsize = params.pop_size
        self.sigma = params.sigma
        self.resolution = params.resolution
        self.trainstep = 0
        self.shape = params.shape
        self.pop = []
        self.bh_data = None

    def act(self, state, idx):
        if idx > 0:
            return self.pop[idx-1].feed(state)

    def policies(self):
        """
        iterator to iterate through pop_size of policies
        """
        self.pop = []
        for i in range(self.popsize):
            network = self.net_type(self.shape)
            if len(self.map) != 0:
                r, other_network = random.choice(list(self.map.values()))
                network.parent(other_network, self.sigma, save_updates=False)
            self.pop.append(network)
            yield network

    def shape_bh(self, bh: np.array):
        """
        returns a behavior array to be used for indexing map

        args:
            bh: behavior metric, np array

        returns 
            bh_new: new, shaped behavior metric, np array
        """
        bh = bh*self.resolution
        bh = bh[[0, 1, 6, 7]]
        if self.bh_data == None:
            self.bh_data = [bh.copy(), bh.copy()]
        else:
            self.bh_data[0] = np.minimum(bh, self.bh_data[0])
            self.bh_data[1] = np.maximum(bh, self.bh_data[1])
        return np.int32(bh).tobytes()

    def train(self, BH: list, R: list) -> None:
        """
        updates map with trainging info

        args:
            BH(list): list of behavior vectors
            R(list): returns list of rewards received 

        return: none
        """
        improves = 0
        for bh, r, agent in zip(BH, R, self.pop):
            bh = self.shape_bh(bh)
            if bh in self.map:
                r_old, agent_old = self.map[bh]
            else:
                r_old = -1e100
            if r > r_old:
                self.map[bh] = [r, agent]
                improves += 1
        return improves
