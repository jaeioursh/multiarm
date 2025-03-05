import numpy as np
from collections import deque
from random import sample

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
		   
	def parent(self,p,sigma,mirror=None):
		self.dw=[]
		self.db=[]
		
		for i in range(len(self.w)):
			
				self.w[i]=p.w[i].copy()
				self.b[i]=p.b[i].copy()
				if mirror is None:
					dw=np.random.normal(0,1,size=self.w[i].shape)
					db=np.random.normal(0,1,size=self.b[i].shape)
				else:
					dw=-mirror.dw[i].copy()
					db=-mirror.db[i].copy()
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
	def __init__(self,params):
		self.parent=net(params.shape)
		self.pop=[net(params.shape) for i in range(params.pop_size)]
		self.hof=deque(maxlen=params.hof_size)
		self.best=net(params.shape)
		self.best_r=None
		self.N=params.pop_size
		self.sigma=params.sigma
		self.LR=params.lr
		self.trainstep=0
		self.shape=params.shape
		self.hof_freq=params.hof_freq
		self.set_parent()
   
	def policies(self):
		return [p.feed for p in self.pop]
   
	def set_parent(self):
		
		for i in range(len(self.pop)):
			indiv=self.pop[i]
			if i%2==0:
				indiv.parent(self.parent,self.sigma)
			else:
				indiv.parent(self.parent,self.sigma,self.pop[i-1])
	def fitshape(self,R):
		if 0:
			R=np.array(R)
			R-=np.mean(R)
			R/=np.std(R)
		else:
			R=np.array(R)
			K=np.argsort(np.argsort(-R))+1
			lmbda=len(K)
			uk=np.log(lmbda/2+1)-np.log(K)
			uk[uk<0]=0
			R=uk/np.sum(uk)-1/lmbda
		return R
	def train(self,R,G):
		if self.best_r is None:
			self.best_r=max(G)
		if max(G)>self.best_r:
			self.best_r=max(G)
			#print(max(R))
			self.best.copy(self.pop[np.argmax(G)])
			self.trainstep=0
			temp=net(self.shape)
			temp.copy(self.best)
			self.hof.append(temp)
		else:
			self.trainstep+=1

		if self.trainstep==self.hof_freq and len(self.hof)>0:
			temp=sample(self.hof,1)[0]
			#self.LR*=0.9
			self.trainstep=0
			self.parent.copy(temp)
		else:
			R=self.fitshape(R)
			
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
 