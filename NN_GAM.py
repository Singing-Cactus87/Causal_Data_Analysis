import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import torch.nn.functional as F


#Dataset 준비

np.random.seed(321)
X1 = np.linspace(0,2*np.pi,200)
np.random.shuffle(X1)
X1_ = np.sin(X1)
X2 = np.linspace(0,2*np.pi,200)
np.random.shuffle(X2)
X2_ = np.cos(X2)
Y = 1*X1_+4*X2_

dt = pd.DataFrame({'x1':X1,'x2':X2,'y':Y})
dt.head()

#Dataset 0과 1 사이로 Min-Max 정규화

from sklearn.preprocessing import MinMaxScaler

mn1 = MinMaxScaler()
dtt =  mn1.fit_transform(dt)
dtt = pd.DataFrame(dtt, columns=dt.columns)



#Custom NN-GAM 구축

class NN_GAM(torch.nn.Module):
  def __init__(self,h1,h2,h3,d):
    super(NN_GAM, self).__init__()
    self.h1 = h1
    self.h2 = h2
    self.h3 = h3
    self.d = d

    class SparseLayer(nn.Linear):
      def __init__(self,input_dim,output_dim,mask,bias=False):
        super().__init__(input_dim,output_dim,bias)
        self.weight = nn.Parameter(self.weight)
        self.register_buffer("mask",mask.T) #Transposed
        self.weight.register_hook(lambda grad: grad * self.mask)

      def forward(self,x):
        w_m = self.weight*self.mask
        return F.linear(x,w_m,self.bias)

    m1 = np.zeros(self.d*self.h1*self.d).reshape(self.d,-1)
    m2 = np.zeros(self.h1*self.d*self.h2*self.d).reshape(self.h1*self.d,-1)
    m3 = np.zeros(self.h2*self.d*self.h3*self.d).reshape(self.h2*self.d,-1)
    m4 = np.zeros(self.h3*self.d*self.d).reshape(self.h3*self.d,-1)

    for i in range(self.d):
      m1[i,(i*self.h1):((i+1)*self.h1)] = 1
      m2[(i*self.h1):((i+1)*self.h1),(i*self.h2):((i+1)*self.h2)] = 1
      m3[(i*self.h2):((i+1)*self.h2),(i*self.h3):((i+1)*self.h3)] = 1
      m4[(i*self.h3):((i+1)*self.h3),i] = 1

    m1 = torch.tensor(m1,dtype=torch.float32);m2 = torch.tensor(m2,dtype=torch.float32);m3 = torch.tensor(m3,dtype=torch.float32);m4 = torch.tensor(m4,dtype=torch.float32)


    self.mask1 = m1
    self.mask2 = m2
    self.mask3 = m3
    self.mask4 = m4

    self.ly1 = SparseLayer(self.d,self.h1*self.d,self.mask1)
    self.o1 = nn.ELU()
    self.ly2 = SparseLayer(self.h1*self.d,self.h2*self.d,self.mask2)
    self.o2 = nn.ELU()
    self.ly3 = SparseLayer(self.h2*self.d, self.h3*self.d,self.mask3)
    self.o3 = nn.ELU()
    self.out = SparseLayer(self.h3*self.d, self.d,self.mask4)
    self.pred = nn.Linear(self.d,1)
    self.pred_out = nn.Identity()

  def forward(self, x):
    o1 = self.o1(self.ly1(x))
    o2 = self.o2(self.ly2(o1))
    o3 = self.o3(self.ly3(o2))
    out = self.out(o3)
    pred1 = self.pred_out(self.pred(out))
    return pred1,out


###### Training

X_ = dtt.iloc[:,0:2].values
Y_ = dtt.iloc[:,-1].values


#Pytorch-based training

torch.manual_seed(321321)
nn_gam1 = NN_GAM(10,40,40,2)
optimizer = torch.optim.Adam(nn_gam1.parameters(),lr=5e-3)

Epochs = 2000; loss_values = []
loss_f = nn.MSELoss()
for i in range(Epochs):
  optimizer.zero_grad()
  y_hat,ot1 = nn_gam1.forward(torch.tensor(X_, dtype=torch.float32))
  loss = loss_f(y_hat,torch.tensor(Y_,dtype=torch.float32).unsqueeze(-1))
  loss.backward()
  optimizer.step()
  loss_values.append(loss)
  if (i+1)%10 ==0: print(i+1, loss)


#Loss Values Visualization

for i in range(2000):
  loss_values[i] = loss_values[i].detach()

plt.plot(loss_values, color="darkred",label="loss_of_NN-GAM")
plt.xlabel("Epochs")
plt.legend()



#GAM result checking

A = np.concatenate((np.linspace(0,1,200).reshape(-1,1),np.linspace(0,1,200).reshape(-1,1)),axis=1)

with torch.no_grad():
  result2,_2 = nn_gam1.forward(torch.tensor(A, dtype=torch.float32))

with torch.no_grad():
  w1 = nn_gam1.pred.weight

w1

#f_1(X_1)
sns.set_style("darkgrid")
plt.plot(np.linspace(0,1,200),_2[:,0]*(-0.5537),color="darkcyan")


#f_2(X_2)
plt.plot(np.linspace(0,1,200),_2[:,1]*(0.5202),color="darkred")



