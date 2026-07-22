# -*- coding: utf-8 -*-
#######################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torch.func import jacrev, vmap
import os
from collections import OrderedDict
import random

import kagglehub

# Download latest version
path = kagglehub.dataset_download("alistairking/weather-long-term-time-series-forecasting")

print("Path to dataset files:", path)#여기서의 반환값을 아래 train_dir에 투입

os.listdir("/root/.cache/kagglehub/datasets/alistairking/weather-long-term-time-series-forecasting/versions/1")

train_dir = "/root/.cache/kagglehub/datasets/alistairking/weather-long-term-time-series-forecasting/versions/1"
dt = pd.read_csv(os.path.join(train_dir, os.listdir(train_dir)[0]))
dt.drop(['date'],axis=1,inplace=True)

dt.head(5)
print(dt.shape)

data = dt.loc[:,['p','T','rh','VPmax']]
data2 = dt.loc[:,['p','T','rh','VPmax']]

X = data.iloc[:,:]
X2 = data2.iloc[:,:]


X_tr1,X_te1,X_tr2,X_te2 = train_test_split(X,X2,test_size=0.2,shuffle=True, random_state=321) #
print(X_tr1.shape,X_te2.shape)

from sklearn.preprocessing import MinMaxScaler

mn1 = MinMaxScaler()
X_tr1 = mn1.fit_transform(X_tr1)
X_tr1 = pd.DataFrame(X_tr1, columns=X.columns)

X_te1 = mn1.transform(X_te1)
X_te1 = pd.DataFrame(X_te1, columns=X.columns)

mn2 = MinMaxScaler()
X_tr2 = mn2.fit_transform(X_tr2)
X_tr2 = pd.DataFrame(X_tr2, columns=X2.columns)

X_te2 = mn2.transform(X_te2)
X_te2 = pd.DataFrame(X_te2, columns=X2.columns)

X_tr1.head(5)

class MADE_custom(torch.nn.Module):
  def __init__(self, X,h):
    super(MADE_custom, self).__init__()
    self.input_dim = X.shape[1]
    self.output_dim = X.shape[1]
    self.h1 = h[0]
    self.h2 = h[1]
    self.h3 = h[2]
    self.h4 = h[3]

    m0 = np.random.permutation(self.input_dim).tolist()
    self.m0 = m0

    M=[]; M.append(m0)
    for l in range(4):
      l_m = []
      for k in range(h[l]):
        ml_k = np.random.randint(low=min(M[l]),high=self.input_dim)
        l_m.append(ml_k)
      M.append(l_m)

    M_total = []
    for l in range(4):
      M_wl = np.zeros(len(M[l])*len(M[l+1])).reshape(len(M[l]),-1)
      for p in range(M_wl.shape[0]):
        for q in range(M_wl.shape[1]):
          if M[l][p] <= M[l+1][q]:
            M_wl[p,q] = 1
      M_total.append(M_wl)

    #M_v = np.zeros(len(M[-1])*self.output_dim).reshape(-1,self.output_dim)
    #for p1 in range(M_v.shape[0]):
    #  for q1 in range(M_v.shape[1]):
    #    if M[-1][p1] < M[0][q1]:
    #      M_v[p1,q1] = 1

    M_v = np.zeros(len(M[-1]) * (self.output_dim * 2)).reshape(-1, self.output_dim * 2)
    for p1 in range(M_v.shape[0]):
      for q1 in range(M_v.shape[1]):
        if M[-1][p1] < M[0][q1 % self.output_dim]:
          M_v[p1, q1] = 1


    M_total.append(M_v)

    class SparseLayer(nn.Linear):
      def __init__(self,input_dim,output_dim,mask,bias=False):
        super().__init__(input_dim,output_dim,bias)
        self.register_buffer("mask",mask.T)
        self.weight.register_hook(lambda grad: grad*self.mask) #CPU

      def forward(self,x):
        w_m = self.weight*self.mask
        return F.linear(x, w_m, self.bias)

    self.model = nn.Sequential(OrderedDict([
        ('en_ly1',SparseLayer(self.input_dim,self.h1,torch.tensor(M_total[0],dtype=torch.float32))),
        ('elu1',nn.ReLU()),
        ('en_ly2',SparseLayer(self.h1,self.h2,torch.tensor(M_total[1],dtype=torch.float32))),
        ('elu2',nn.ReLU()),
        ('en_ly3',SparseLayer(self.h2,self.h3,torch.tensor(M_total[2],dtype=torch.float32))),
        ('elu3',nn.ReLU()),
        ('en_ly4',SparseLayer(self.h3,self.h4,torch.tensor(M_total[3],dtype=torch.float32))),
        ('elu4',nn.ReLU()),
        ('en_out',SparseLayer(self.h4,self.output_dim*2,torch.tensor(M_total[4],dtype=torch.float32))),
        ('Id1',nn.Identity()),
    ]))



  def forward(self, X):
    X_hat = self.model(X)

    return X_hat

made1 = MADE_custom(X=torch.tensor(np.array(X_tr1),dtype=torch.float32),h=[20,30,30,20])

made1(X=torch.tensor(np.array(X_tr1),dtype=torch.float32))

class ANF(torch.nn.Module):
  def __init__(self,X_,h_):
    super(ANF, self).__init__()
    self.h_ = h_
    self.mu_alpha_NN = MADE_custom(X=X_,h=h_)
    self.order = np.argsort(self.mu_alpha_NN.m0)
    self.input_dim = X_.shape[1]

  def _get_mu_alpha(self, x):
    out = self.mu_alpha_NN(x)
    mu, alpha = torch.split(out, self.input_dim, dim=1)
    return mu, alpha

  def forward(self,x):
    mu, alpha = self._get_mu_alpha(x)
    u = (x-mu)*torch.exp(-alpha)
    return u, alpha

  def loss_func(self,x):
    u_hat, _ = self.forward(x)
    mu, alpha = self._get_mu_alpha(x)
    loss = torch.sum((0.5*torch.sum(u_hat**2,dim=1)+torch.sum(alpha,dim=1)))/x.shape[0]-np.log(1/np.sqrt(2*np.pi))*x.shape[1]
    return loss

  def inverse_sample(self,u):
    x = torch.zeros_like(u)

    for idx in self.order:
      with torch.no_grad():
        mu, alpha = self._get_mu_alpha(x)

        x[:, idx] = torch.exp(alpha[:, idx]) * u[:, idx] + mu[:, idx]

    return x

class MAF(torch.nn.Module):
  def __init__(self,X,h):
    super(MAF, self).__init__()
    self.ANF1 = ANF(X_=X,h_=h)
    self.ANF2 = ANF(X_=X,h_=h)
    self.ANF3 = ANF(X_=X,h_=h)
    self.ANF4 = ANF(X_=X,h_=h)
    self.ANF5 = ANF(X_=X,h_=h)

  def forward(self,x):
    x, alpha1 = self.ANF1(x)
    x, alpha2 = self.ANF2(x)
    x, alpha3 = self.ANF3(x)
    x, alpha4 = self.ANF4(x)
    u, alpha5 = self.ANF5(x)
    return u, alpha1, alpha2, alpha3, alpha4, alpha5

  def loss_func(self,x):
    u_hat, alpha1, alpha2, alpha3, alpha4, alpha5 = self.forward(x)
    loss = torch.sum((0.5*torch.sum(u_hat**2,dim=1)+torch.sum(alpha1,dim=1)+torch.sum(alpha2,dim=1)+torch.sum(alpha3,dim=1)+torch.sum(alpha4,dim=1)+torch.sum(alpha5,dim=1)))/x.shape[0]-np.log(1/np.sqrt(2*np.pi))*x.shape[1]
    return loss

  def inverse_sample(self, u):
    x = self.ANF5.inverse_sample(u)
    x = self.ANF4.inverse_sample(x)
    x = self.ANF3.inverse_sample(x)
    x = self.ANF2.inverse_sample(x)
    x = self.ANF1.inverse_sample(x)


    return x

torch.manual_seed(678)
np.random.seed(321)
Epochs = 250
maf = MAF(X=torch.tensor(np.array(X_tr1),dtype=torch.float32),h=[20,30,30,20])

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        #torch.nn.init.zeros_(m.bias)

maf.apply(init_weights)

i = 0
loss_t = []

optimizer = torch.optim.Adam(maf.parameters(),lr=0.001) #0.001

while i < Epochs:
  optimizer.zero_grad()
  loss = maf.loss_func(torch.tensor(np.array(X_tr1),dtype=torch.float32))
  loss.backward()
  optimizer.step()
  loss_t.append(loss)
  i += 1
  if i%10 == 0: print(i,loss)

sns.set_style("darkgrid")
for i in range(Epochs):
  loss_t[i] = loss_t[i].detach()

plt.plot(loss_t, color="darkred",label="loss_of_MAF")
plt.xlabel("Epochs")
plt.legend();

torch.manual_seed(1234)
with torch.no_grad():
  x_samp = maf.inverse_sample(torch.randn(1000, maf.ANF1.input_dim)) #100

x_samp[:10,:]

X_tr1.iloc[:10,:]

cleaned_arr_1 = np.array(x_samp)[:,0][np.isfinite(np.array(x_samp)[:,0])]
cleaned_arr_1=cleaned_arr_1[(cleaned_arr_1 < np.quantile(cleaned_arr_1,q=0.9)) & (cleaned_arr_1 > np.quantile(cleaned_arr_1,q=0.1))]

sns.kdeplot(X_tr1.iloc[:,0],color="red",label="true",fill=True)
sns.kdeplot(cleaned_arr_1,color="blue",label="generated",fill=True)
plt.legend();

cleaned_arr_2 = np.array(x_samp)[:,1][np.isfinite(np.array(x_samp)[:,1])]
cleaned_arr_2=cleaned_arr_2[(cleaned_arr_2 < np.quantile(cleaned_arr_2,q=0.9)) & (cleaned_arr_2 > np.quantile(cleaned_arr_2,q=0.1))]

sns.kdeplot(X_tr1.iloc[:,1],color="red",label="true",fill=True)
sns.kdeplot(cleaned_arr_2,color="blue",label="generated",fill=True)
plt.legend();

cleaned_arr_3 = np.array(x_samp)[:,2][np.isfinite(np.array(x_samp)[:,2])]
cleaned_arr_3=cleaned_arr_3[(cleaned_arr_3 < np.quantile(cleaned_arr_3,q=0.9)) & (cleaned_arr_3 > np.quantile(cleaned_arr_3,q=0.1))]

sns.kdeplot(X_tr1.iloc[:,2],color="red",label="true",fill=True)
sns.kdeplot(cleaned_arr_3,color="blue",label="generated",fill=True)
plt.legend();

cleaned_arr_4 = np.array(x_samp)[:,3][np.isfinite(np.array(x_samp)[:,3])]
cleaned_arr_4=cleaned_arr_4[(cleaned_arr_4 < np.quantile(cleaned_arr_4,q=0.9)) & (cleaned_arr_4 > np.quantile(cleaned_arr_4,q=0.1))]

sns.kdeplot(X_tr1.iloc[:,3],color="red",label="true",fill=True)
sns.kdeplot(cleaned_arr_4,color="blue",label="generated",fill=True)
plt.legend();

