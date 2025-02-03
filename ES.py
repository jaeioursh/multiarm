import numpy as np

 

class net:

    def __init__(self,s):

        self.shape=s

        self.depth=len(s)-1

        self.dw=[]

        self.db=[]

       

        self.shuffle()

        self.e=None

       

       

       

    def shuffle(self):

        s=self.shape

        self.w=[np.random.normal(0,0.05,size=[s[i],s[i+1]]) for i in range(len(s)-1)]

        self.b=[np.random.normal(0,0.05,size=[1,s[i+1]]) for i in range(len(s)-1)]

 

    def copy(self,p):

        for i in range(len(self.w)):

            self.w[i]=p.w[i].copy()

            self.b[i]=p.b[i].copy()

           

    def parent(self,p,sigma):

        self.dw=[]

        self.db=[]

        for i in range(len(self.w)):

            self.w[i]=p.w[i].copy()

            self.b[i]=p.b[i].copy()

 

            dw=np.random.normal(0,1,size=self.w[i].shape)

            db=np.random.normal(0,1,size=self.b[i].shape)

             

            self.w[i]+=sigma*dw

            self.b[i]+=sigma*db

 

            self.dw.append(dw)

            self.db.append(db)

           

 

    def h(self,x):

        return np.tanh(x)

 

   

    def feed(self,x):

       

        for w,b in zip(self.w,self.b):

            x=self.h(np.matmul(x,w)+b)

        return x[0]

   


 

    def softmax(self,x):

        return np.exp(x)/sum(np.exp(x))

   

 

class ES:

    def __init__(self,shape,pop_size,lr=0.01,sigma=0.001):

        self.parent=net(shape)

        self.pop=[net(shape) for i in range(pop_size)]

        self.best=net(shape)

        self.best_r=-1e20

        self.N=pop_size

        self.sigma=sigma

        self.LR=lr

        self.set_parent()

   

    def policies(self):

        return [p.feed for p in self.pop]

   

    def set_parent(self):

        for indiv in self.pop:

            indiv.parent(self.parent,self.sigma)

 

    def train(self,R):

        if max(R)>self.best_r:

            self.best_r=max(R)

            self.best.copy(self.pop[np.argmax(R)])

            #print(self.best_r)

        R=np.array(R)

        R-=np.mean(R)
        R/=np.std(R)
        
        dw=[np.zeros_like(w) for w in self.parent.w]

        db=[np.zeros_like(b) for b in self.parent.b]

        for r,indiv in zip(R,self.pop):

            for i in range(len(dw)):

                dw[i]+=r*indiv.dw[i]

                db[i]+=r*indiv.db[i]

        for i in range(len(dw)):

            dw[i]/=self.N*self.sigma

            db[i]/=self.N*self.sigma

           

            self.parent.w[i]+=self.LR*dw[i]

            self.parent.b[i]+=self.LR*db[i]

        self.set_parent()

 