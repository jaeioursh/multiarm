
import numpy as np
from ES import ES, MultiHeadNetRes


class MAES:
    def __init__(self, params):
        self.params = params
        self.n_agents = params.n_agents
        self.popsize = params.pop_size
        self.agents = [ES(params) for i in range(params.n_agents)]

    def train(self, R, G):
        for r, agent in zip(R, self.agents):
            agent.train(r, G)

    def act(self, states, idx):
        action = []
        for state, agent in zip(states, self.agents):
            if idx > 0:
                action.append(agent.pop[idx-1].feed(state))
            elif idx == 0:
                action.append(agent.best.feed(state))
            else:
                action.append(agent.hof[abs(idx)-1].feed(state))
        return np.array(action)

# MultiHead Evolution Strategy


class MHES:
    def __init__(self, params):
        self.params = params
        self.popsize = params.pop_size
        self.agent = ES(params)

    def train(self, R):
        self.agent.train(R)

    def act(self, state, idx):
        if idx > 0:
            return self.agent.pop[idx-1].feed(state)
        elif idx == 0:
            return self.agent.best.feed(state)
        else:
            return self.agent.hof[abs(idx)-1].feed(state)
