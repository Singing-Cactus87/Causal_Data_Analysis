import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Mask_return():
  def __init__(self, D,K):
    super(Mask_return,self).__init__()
    self.D = D
    s = []; s.append(self.D)
    self.K = s+K #input 노드 개수도 추가
    self.L = len(self.K) #초기 input len(K)+1

  def sampling_m(self):
    min_m = 1 #initial value
    Md = np.sort(np.arange(1,self.D+1))[::-1] #Fixed for explanation. Should Use Random Shuffling in real analysis
    m_ = []; m_.append(Md)

    for l in range(1,self.L,1):
      m_l = []
      for k in range(self.K[l]):
        m_lk = int(np.random.randint(low=min_m,high=self.D,size=1)[0])
        m_l.append(m_lk)
      min_m = np.min(m_l)
      m_.append(m_l)

    return m_

  def return_mask(self):
    m = self.sampling_m()
    M_ = []
    for l in range(1,self.L,1):
      M_L = np.zeros(self.K[l-1]*self.K[l]).reshape(-1,self.K[l])
      for i in range(self.K[l-1]):
        for j in range(self.K[l]):
          if m[l][j] >= m[l-1][i]:
            M_L[i,j] = 1
          else:
            M_L[i,j] = 0
      M_.append(M_L)

    Md = np.sort(np.arange(1,self.D+1))[::-1]
    M_V = np.zeros(self.K[self.L-1]*self.D).reshape(-1,self.D)
    for i in range(self.K[self.L-1]):
      for j in range(self.D):
        if Md[j] > m[self.L-1][i]:
          M_V[i,j] = 1
        else:
          M_V[i,j] = 0
    M_.append(M_V)

    return M_, m



Mask_f = Mask_return(3,[4,4,4])

np.random.seed(321)
Mask_set,node = Mask_f.return_mask()

np.where(Mask_set[0]@Mask_set[1]@Mask_set[2]@Mask_set[3]>0,1,0)

node



sns.heatmap(Mask_set[0],cmap="gray",linewidths=1,linecolor="black")
sns.heatmap(Mask_set[1],cmap="gray",linewidths=1,linecolor="black")
sns.heatmap(Mask_set[2],cmap="gray",linewidths=1,linecolor="black")
sns.heatmap(Mask_set[3],cmap="gray",linewidths=1,linecolor="black")
