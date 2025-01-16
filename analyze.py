import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

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

    aln,mat=align(x,y)
    plt.subplot(1,2,1)
    plt.imshow(mat)
    plt.subplot(1,2,2)
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

    plt.plot(t,x.numpy())
    plt.plot(t,y.numpy())
    plt.plot(t,y.numpy()+w.detach().numpy())
    plt.plot(t,w.detach().numpy(),":")
    plt.legend(["Global","Local","Local Shaped","Shaping Fn."])
    plt.show()

if __name__ == "__main__":
    toy_test()