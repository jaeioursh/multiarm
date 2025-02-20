import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from alignment_network import alignment_network
def train_test():
    with open("logs/data.dat","rb") as f:
        data=pkl.load(f)
        
    S,G,R=[], [], []
    for s,g,r in data:
        S.append(s)
        G.append(g)
        R.append(r)
    S,G,R=np.array(S),np.array(G),np.array(R)

    print(np.mean(G),max(G),len(G))

def align(x,y):
    rank=len(x)
    dx=x-x.T
    dy=y-y.T
    diff=np.multiply(dx,dy)
    diff[diff>0]=1
    diff[diff==0]=0.5
    diff[diff<0]=0.0
    diff[np.eye(rank).astype(bool)]=0.0
    aln = np.sum(diff)/(rank*rank-rank)
    return aln,diff

sig=torch.nn.Sigmoid()

def alignment_loss(o, t):   
        O=o-torch.transpose(o,0,1)
        T=t-torch.transpose(t,0,1)
        align = torch.mul(O,T)
        align = sig(align)
        loss = -torch.mean(align)#,dim=0)
        return loss
    
def toy_test():
    t=np.linspace(-1,1,20).reshape((-1,1))
    x=t*t 
    x[t>0]=0
    y=t 
    torch.manual_seed(0)
    aln,mat=align(x,y)
    #plt.subplot(1,2,1)
    #plt.imshow(mat)
    #plt.subplot(1,2,2)
    x=torch.from_numpy(x).type(torch.float32)
    y=torch.from_numpy(y).type(torch.float32)
    w = torch.autograd.Variable(torch.randn(t.shape), requires_grad=True)
    mse=torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam([w], lr=0.001)
    for i in range(10000):
        optimizer.zero_grad()
        loss = alignment_loss(x+torch.randn(t.shape)/10,w+y)+ 0.05*torch.mean(torch.square(w))
        loss.backward()
        optimizer.step()

    plt.plot(t,x.numpy(),label="Global")
    plt.plot(t,y.numpy(),label="Local")
    plt.plot(t,w.detach().numpy(),":",label="Shaping Fn.")
    plt.plot(t,y.numpy()+w.detach().numpy(),label="Local Shaped")
    
    plt.xlabel("State")
    plt.ylabel("Reward")
    plt.ylim(-1,2.5)
    plt.legend()
    plt.show()

def two_step():
    x=np.linspace(0.5,2.5)
    val=3.1
    f1=lambda X: X**1.3
    f2=lambda X: -2.8*X**-2.8
    f3=lambda X: -np.log(2.8*X**-2.8)
    
    f1=np.vectorize(f1)
    f2=np.vectorize(f2)
    f3=np.vectorize(f3)

    plt.subplot(1,2,1)
    plt.title("f(x)")
    for f in [f1,f2,f3]:
        plt.plot(x,f(x))

    plt.subplot(1,2,2)
    plt.title("f(x) + f(3-x)")
    for f in [f1,f2,f3]:
        y=f(x)+f(val-x)
        print(np.argmax(y))
        plt.plot(x,y)
    plt.show()


def three_step():
    f1=lambda X: X**1.3
    f2=lambda X: -2.8*X**-2.8
    f3=lambda X: -np.log(-f2(X))

    f1=np.vectorize(f1)
    f2=np.vectorize(f2)
    f3=np.vectorize(f3)
    idx=0
    funcs=[f1,f2,f3]
    for f in funcs:
        low,high,res=0.1,29000,300
        idx+=1
        x=np.linspace(low,high,res)
        y=np.linspace(low,high,res)
        xx,yy=np.meshgrid(x,y)
        zz=-xx-yy+30000
        zz[zz<=0]=np.NAN
        print(zz.dtype)
        val=f(xx)+0.9*f(yy)+0.9**2*f(zz)
        print(min)
        val=np.nan_to_num(val,nan=np.nanmin(val))
        print(np.unravel_index(val.argmax(), val.shape),val.max(),val.min())
        plt.subplot(1,len(funcs),idx)
        plt.imshow(val)

    plt.show()

class prms():
    def __init__(self):
        self.device="cpu"
        self.state_dim=26
        self.aln_hidden=64
        self.aln_lr=0.0003
        self.aln_batch=16
        self.aln_train_steps=1
        self.deq_len=100

        self.use_l1=True
        self.use_l2=False
        self.l_penalty=0.1
            
def alg_test(): 
    agent_idx=1
    params=prms()
    net=alignment_network(params)
    fidx=6
    with open("logs/train_data"+str(fidx)+".dat","rb") as f:
        info = pkl.load(f)
    for state,G,reward,done in info:
        net.add_sample(reward[agent_idx],G,state[agent_idx],done=done)
    L=[]
    for i in range(4000):
        l=net.train()
        print(i,l[0])
        L.append(l)
    L=np.array(L).T
    N=4
    for i in range(N):
        plt.subplot(1,N,i+1)
        plt.plot(L[i])
    plt.legend(["Alignment Loss","Avg. Shaping","Std. Shaping", "Max Shaping"])
    plt.show()


if __name__ == "__main__":
    alg_test()