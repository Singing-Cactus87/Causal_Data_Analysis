# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torch.func import jacrev, vmap
from collections import OrderedDict

pip install bnlearn

import bnlearn as bn

class Generalized_DAG_GNN(torch.nn.Module):
  def __init__(self, input_dim,lat_dim,h=[20,20]):
    super(Generalized_DAG_GNN,self).__init__()
    self.input_dim = input_dim
    self.lat_dim = lat_dim
    self.h1 = h[0]*lat_dim
    self.h2 = h[1]*lat_dim

    class SparseLayer(nn.Linear):
      def __init__(self,input_dim,output_dim,mask,bias=False):
        super().__init__(input_dim,output_dim,bias)
        self.register_buffer("mask",mask.T) #Transposed
        self.weight.register_hook(lambda grad: grad * self.mask)

      def forward(self,x):
        w_m = self.weight*self.mask
        return F.linear(x,w_m,self.bias)

    class AdjLayer(nn.Linear):
      def __init__(self,input_dim,output_dim,mask,bias=False):
        super().__init__(input_dim,output_dim,bias)
        self.register_buffer("mask",mask.T) #Transposed
        self.weight.register_hook(lambda grad: grad * self.mask)

      def forward(self,x):
        out = self.weight*self.mask
        return F.linear(x,out,self.bias)

    #X = f(A^T X)

    mat1 = np.zeros(self.input_dim*self.h1).reshape(self.input_dim,self.h1)
    mat2 = np.zeros(self.h1*self.h2).reshape(self.h1,self.h2)
    mat3 = np.zeros(self.h2*self.lat_dim).reshape(self.h2,self.lat_dim)
    mat4 = np.ones(self.lat_dim*self.lat_dim).reshape(self.lat_dim,self.lat_dim)
    mat5 = np.zeros(self.lat_dim*(self.lat_dim+self.lat_dim)).reshape(self.lat_dim,self.lat_dim+self.lat_dim)
    mat6 = np.zeros(self.h2*self.lat_dim).reshape(self.h2,self.lat_dim).T
    mat7 = np.zeros(self.h1*self.h2).reshape(self.h1,self.h2).T
    mat8 = np.zeros(self.input_dim*self.h1).reshape(self.input_dim,self.h1).T
    for i in range(self.lat_dim):
      mat1[i,(i*h[0]):((i+1)*h[0])] = 1
      mat2[(i*h[0]):((i+1)*h[0]),(i*h[1]):((i+1)*h[1])] = 1
      mat3[(i*h[1]):((i+1)*h[1]),i] = 1
      mat4[i,i] = 0
      mat5[i,i]= 1
      mat5[i,i+self.lat_dim] = 1
      mat6[i,(i*h[1]):((i+1)*h[1])] = 1
      mat7[(i*h[1]):((i+1)*h[1]),(i*h[0]):((i+1)*h[0])] = 1
      mat8[(i*h[0]):((i+1)*h[0]),i] = 1

    self.mask1 = torch.tensor(mat1, dtype=torch.float32)
    self.mask2 = torch.tensor(mat2, dtype=torch.float32)
    self.mask3 = torch.tensor(mat3, dtype=torch.float32)
    self.mask4 = torch.tensor(mat4, dtype=torch.float32)
    self.mask5 = torch.tensor(mat5, dtype=torch.float32)
    self.mask6 = torch.tensor(mat6, dtype=torch.float32)
    self.mask7 = torch.tensor(mat7, dtype=torch.float32)
    self.mask8 = torch.tensor(mat8, dtype=torch.float32)




    self.ADJ = AdjLayer(self.lat_dim,self.lat_dim,self.mask4)


    self.ENCODER = nn.Sequential(OrderedDict([
        ('en_ly1',SparseLayer(self.input_dim,self.h1,self.mask1)),
        ('relu1',nn.ReLU()),
        ('en_ly2',SparseLayer(self.h1,self.h2,self.mask2)),
        ('relu2',nn.ReLU()),
        ('en_out',SparseLayer(self.h2,self.lat_dim,self.mask3)),
        ('Id1',nn.Identity()),
    ]))


    self.DECODER = nn.Sequential(OrderedDict([
        ('dec_ly1',SparseLayer(self.lat_dim,self.h2,self.mask6)),
        ('relu1',nn.ReLU()),
        ('dec_ly2',SparseLayer(self.h2,self.h1,self.mask7)),
        ('relu2',nn.ReLU()),
        ('dec_out',SparseLayer(self.h1,self.input_dim,self.mask8)),
        ('Id2',nn.Identity()),
    ]))



    self.Enc_v1 = nn.Sequential(OrderedDict([
        ('enc_v1',SparseLayer(self.lat_dim,self.lat_dim+self.lat_dim,self.mask5)),
        ('Id2',nn.Identity()),
    ]))

  def reparam(self,x):
    ec = self.Enc_v1(x)
    mean, lv = torch.split(ec,split_size_or_sections=self.lat_dim,dim=-1)
    eps = torch.randn_like(mean)
    return eps*torch.exp(lv*0.5) + mean



  def enc(self,x):
    a1 = self.ENCODER(x)
    I = torch.eye(self.lat_dim, device=x.device)
    M = I-torch.transpose(self.ADJ.weight,0,1)
    Z_1 = torch.matmul(a1,M)
    Z = self.reparam(Z_1)
    return Z


  def dec(self,x,sigmoid=False):
    x_T = torch.transpose(x,0,1)
    I = torch.eye(self.lat_dim, device=x.device)
    M = I-torch.transpose(self.ADJ.weight,0,1)
    a2_T = torch.linalg.solve(M, x_T)
    a2 = torch.transpose(a2_T,0,1)
    Z_2 = self.DECODER(a2)
    X_hat = self.reparam(Z_2)
    if sigmoid == True:
      X_hat = torch.sigmoid(X_hat)
    return X_hat


  def forward(md,x):
    z = md.enc(x)
    x_hat = md.dec(z,sigmoid=True)
    return x_hat



import bnlearn as bn

np.random.seed(123) # 123, 333, 456
dg = bn.import_DAG("sprinkler")
df = bn.sampling(dg, n=200)

#dg

true_dag = bn.dag2adjmat(dg['model'])*1

true_dag

torch.manual_seed(678)
Epochs = 1000

lat_dim_ = 4
G_D = Generalized_DAG_GNN(lat_dim_,lat_dim_,[5,5])

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        #torch.nn.init.zeros_(m.bias)

G_D.apply(init_weights)
torch.nn.init.zeros_(G_D.ADJ.weight)

alpha=0.6
i = 0
rho = 0.1
gamma=0.9
beta = 1.01
lamb = 0.1 #L1-regularization

loss_ms = []
loss_t = []


adj_params = [G_D.ADJ.weight]
other_params = [i for _, i in G_D.named_parameters() if "ADJ" not in _]

optimizer1 = torch.optim.Adam(adj_params,lr=0.01)
optimizer2 = torch.optim.Adam(other_params,lr=0.004)
loss_f = nn.MSELoss()
dff = torch.tensor(np.array(df), dtype=torch.float32)

while i < Epochs:
  optimizer1.zero_grad()
  optimizer2.zero_grad()
  x_hat = G_D.forward(dff)
  loss1 = loss_f(x_hat,dff)+0.5*torch.sum(torch.square(G_D.enc(dff)))*0.00005;loss_ms.append(loss1)
  h_a = torch.trace(torch.linalg.matrix_exp(torch.mul(G_D.ADJ.weight,G_D.ADJ.weight)))-lat_dim_
  loss_total = loss1+alpha*h_a+rho*0.5*torch.abs(h_a)**2+lamb*torch.linalg.norm(G_D.ADJ.weight,ord=1);loss_t.append(loss_total)
  loss_total.backward()
  optimizer1.step()
  optimizer2.step()
  with torch.no_grad():
    h_a_new = torch.trace(torch.linalg.matrix_exp(torch.mul(G_D.ADJ.weight,G_D.ADJ.weight)))-lat_dim_
    alpha =  alpha + rho * h_a_new
  if (torch.abs(h_a_new) >= gamma*torch.abs(h_a)):
    rho = beta*rho
  else:
    rho = rho

  i += 1
  if i%10 == 0: print(i,loss_total)

G_D.ADJ.weight[:,:].T

sns.set_style("darkgrid")
for i in range(Epochs):
  loss_t[i] = loss_t[i].detach()
plt.plot(loss_t,color="darkred",label="total loss")
plt.legend()

with torch.no_grad():
  AD = G_D.ADJ.weight.detach()
sns.heatmap(np.array(AD).T, cmap="vlag",center=0)
plt.show()

sns.heatmap(np.where(np.abs(np.array(AD).T)>np.quantile(np.abs(np.array(AD).T),0.75),1,0), cmap="gray_r",linewidths=1,linecolor="black")
plt.show()

sns.heatmap(true_dag, cmap="gray_r",linewidths=1, linecolor="black")

model_hc = bn.structure_learning.fit(df, methodtype='hc',scoretype='bic')

model_hc['adjmat']*1

sns.heatmap(model_hc['adjmat']*1,cmap="gray_r",linewidths=1,linecolor="black")

sns.heatmap(true_dag,cmap="gray_r",linewidths=1,linecolor="black")

shd1 = np.sum(np.sum(np.abs(model_hc['adjmat']*1-true_dag)));shd1

shd2 = np.sum(np.sum(np.abs(np.where(np.abs(np.array(AD).T)>np.quantile(np.abs(np.array(AD).T),0.75),1,0)-true_dag)));shd2

model_hc2 = bn.structure_learning.fit(df, methodtype='direct-lingam')

model_hc2['adjmat']*1

