import numpy as np
from collections import deque
from random import sample


class Net:
    def __init__(self, shape):
        self.shape = shape
        self.dw = []
        self.db = []

        self.shuffle()
        self.e = None

    def shuffle(self):
        s = self.shape
        self.w = [np.random.normal(0, 0.05, size=[s[i], s[i+1]])
                  for i in range(len(s)-1)]
        self.b = [np.random.normal(0, 0.05, size=[1, s[i+1]])
                  for i in range(len(s)-1)]

    def copy(self, p):
        for i in range(len(self.w)):
            self.w[i] = p.w[i].copy()
            self.b[i] = p.b[i].copy()

    def parent(self, p, sigma, mirror=None, save_updates=True):
        self.dw = []
        self.db = []

        for i in range(len(self.w)):

            self.w[i] = p.w[i].copy()
            self.b[i] = p.b[i].copy()
            if mirror is None:
                dw = np.random.normal(0, 1, size=self.w[i].shape)
                db = np.random.normal(0, 1, size=self.b[i].shape)
            else:
                dw = -mirror.dw[i].copy()
                db = -mirror.db[i].copy()
            self.w[i] += sigma*dw
            self.b[i] += sigma*db
            if save_updates:
                self.dw.append(dw)
                self.db.append(db)

    def h(self, x):
        return np.tanh(x)

    def relu(self, x):
        x = np.copy(x)
        x[x < 0] *= 0.1
        return x

    def feed(self, x):

        for w, b in zip(self.w, self.b):
            x = self.h(np.matmul(x, w)+b)
        return x[0]

    def softmax(self, x):
        return np.exp(x)/sum(np.exp(x))


class MultiHeadNet(Net):
    def shuffle(self):
        '''shuffles/generates new network params, ws and bs

        args: 
            params.shape : trunk_shape, n_heads, head_shape

        returns:
                none
        '''
        trunk_shape, n_heads, head_shape = self.shape
        self.trunk_shape = trunk_shape
        self.n_heads = n_heads
        self.head_shape = head_shape

        self.w = []
        self.b = []
        # add trunk layers
        for i in range(len(trunk_shape)-1):
            self.w.append(np.random.normal(
                0, 0.05, size=[trunk_shape[i], trunk_shape[i+1]]))
            self.b.append(np.random.normal(
                0, 0.05, size=[1, trunk_shape[i+1]]))
        # add head layers
        for j in range(n_heads):
            for i in range(len(head_shape)-1):
                self.w.append(np.random.normal(
                    0, 0.05, size=[head_shape[i], head_shape[i+1]]))
                self.b.append(np.random.normal(
                    0, 0.05, size=[1, head_shape[i+1]]))

    def feed(self, x):
        """ feeds input (x) into network and returns outputs
        args:
            x (numpy array): input state variables

        returns:
            outs (np array): output of each head
        """
        idx = 0
        # feed input to trunk and get trunk output
        for i in range(len(self.trunk_shape)-1):
            x = self.h(np.matmul(x, self.w[idx]) + self.b[idx])
            idx += 1
        trunk_out = x
        # get output of each trunk, given head
        outs = []
        for j in range(self.n_heads):
            x = trunk_out
            for i in range(len(self.head_shape)-1):
                x = self.h(np.matmul(x, self.w[idx]) + self.b[idx])
                idx += 1
            outs.append(x)
        return np.array(outs)


class MultiHeadNetRes(MultiHeadNet):
    def __init__(self, shape):
        shape = [s for s in shape]
        shape[1] *= 2  # double number of heads (high res/low res)
        super().__init__(shape)

    def feed(self, x):
        outs = super().feed(x)
        n_heads = self.n_heads//2
        low = outs[:n_heads]
        high = outs[n_heads:]
        high *= 0.7
        low = np.round(low*10)/10
        return low+high


class ES:
    def __init__(self, params):
        self.net_type = params.net_type
        self.parent = self.net_type(params.shape)
        self.pop = [self.net_type(params.shape)
                    for i in range(params.pop_size)]
        self.hof = deque(maxlen=params.hof_size)
        self.best = self.net_type(params.shape)
        self.best_r = None
        self.N = params.pop_size
        self.sigma = params.sigma
        self.LR = params.lr
        self.trainstep = 0
        self.shape = params.shape
        self.hof_freq = params.hof_freq
        self.set_parent()

    def policies(self):
        return [p.feed for p in self.pop]

    def set_parent(self):

        for i in range(len(self.pop)):
            indiv = self.pop[i]
            if i % 2 == 0:
                indiv.parent(self.parent, self.sigma)
            else:
                indiv.parent(self.parent, self.sigma, self.pop[i-1])

    def fitshape(self, R):
        if 0:
            R = np.array(R)
            R -= np.mean(R)
            R /= np.std(R)
        else:
            R = np.array(R)
            K = np.argsort(np.argsort(-R))+1
            lmbda = len(K)
            uk = np.log(lmbda/2+1)-np.log(K)
            uk[uk < 0] = 0
            R = uk/np.sum(uk)-1/lmbda
        return R

    def train(self, R):
        if self.best_r is None:
            self.best_r = max(R)
        if max(R) > self.best_r:
            self.best_r = max(R)
            # print(max(R))
            self.best.copy(self.pop[np.argmax(R)])
            self.trainstep = 0
            temp = self.net_type(self.shape)
            temp.copy(self.best)
            self.hof.append(temp)
        else:
            self.trainstep += 1

        if self.trainstep == self.hof_freq and len(self.hof) > 0:
            temp = sample(self.hof, 1)[0]
            # self.LR*=0.9
            self.trainstep = 0
            self.parent.copy(temp)

        else:
            R = self.fitshape(R)

            dw = [np.zeros_like(w) for w in self.parent.w]
            db = [np.zeros_like(b) for b in self.parent.b]
            for r, indiv in zip(R, self.pop):
                for i in range(len(dw)):
                    dw[i] += r*indiv.dw[i]
                    db[i] += r*indiv.db[i]
            for i in range(len(dw)):
                dw[i] /= self.N*self.sigma
                db[i] /= self.N*self.sigma

                self.parent.w[i] += self.LR*dw[i]
                self.parent.b[i] += self.LR*db[i]
        self.set_parent()
