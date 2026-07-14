import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import os
from collections import OrderedDict
import random

class MADE_custom(torch.nn.Module):
  def __init__(self, X,h):
    super(MADE_custom, self).__init__()
    self.input_dim = X.shape[1]
    self.output_dim = X.shape[1]
    self.h1 = h[0]
    self.h2 = h[1]
    self.h3 = h[2]
    self.h4 = h[3]

    #
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

    M_v = np.zeros(len(M[-1])*self.output_dim).reshape(-1,self.output_dim)
    for p1 in range(M_v.shape[0]):
      for q1 in range(M_v.shape[1]):
        if M[-1][p1] < M[0][q1]:
          M_v[p1,q1] = 1



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
        ('en_out',SparseLayer(self.h4,self.output_dim,torch.tensor(M_total[4],dtype=torch.float32))),
        ('Id1',nn.Identity()),
    ]))



  def forward(self, X):
    X_hat = self.model(X)

    return X_hat
