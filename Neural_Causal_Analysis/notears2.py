# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras import Sequential

pip install bnlearn

import bnlearn as bn

#Importing Causal Dataset
np.random.seed(321)
dg = bn.import_DAG("sprinkler")
df = bn.sampling(dg, n=200)

true_dag = bn.dag2adjmat(dg['model'])*1; true_dag


#Defining Generalized Custom NoTears

class Generalized_NoTears(tf.keras.Model):
  def __init__(self, input_dim,lat_dim,h=[20,20]):
    super(Generalized_NoTears,self).__init__()
    self.input_dim = input_dim
    self.lat_dim = lat_dim
    self.h1 = h[0]
    self.h2 = h[1]

    #X = f(A^T X)

    self.ADJ = Sequential([
        InputLayer(shape=(self.input_dim,)),
        Dense(self.lat_dim,activation="linear",kernel_initializer="zeros",use_bias=False)
    ])

    self.ENCODER = Sequential([
        InputLayer(shape=(1,)),
        Dense(self.h1,activation="linear",kernel_initializer="glorot_normal",use_bias=False),
        Dense(self.h2,activation="linear",kernel_initializer="glorot_normal",use_bias=False),
        Dense(1,activation="sigmoid")
    ])


  def enc(self,x):
    z = self.ADJ(x)
    Z = tf.zeros((x.shape[0],1))
    for i in range(self.lat_dim):
      Z1 = z[:,i]
      Z1 = tf.reshape(Z1,shape=[x.shape[0],1])
      Z2 = self.ENCODER(Z1)
      Z = tf.concat([Z,Z2],1)
    x_hat = Z[:,1:]
    return x_hat

  def mse_loss_gnt(md,x):
    x_hat = md.enc(x)
    x_ = tf.convert_to_tensor(x)
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(x_,x_hat)
    return loss

#Ex
NT_G = Generalized_NoTears(4,4,h=[32,16])

#Initial weight (ADJ)
NT_G.ADJ.weights[0]

NT_G.mse_loss_gnt(np.array([[1,2,4,2],[1,3,2,1]]))

#Training Generalized NoTears

tf.keras.utils.set_random_seed(789) #


Epochs = 500
lat_dim_ = 4
NT_G1 = Generalized_NoTears(df.shape[1],lat_dim_,h=[16,16])

alpha=0.6
i = 0
rho = 0.25 #
gamma=0.9
beta = 1.2 #
lamb = 0.1 #L1-regularization


loss_ms = []
loss_t = []
basic_opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
basic_opt2 = tf.keras.optimizers.Adam(learning_rate=0.001)

while i < Epochs:
    with tf.GradientTape() as dv_1, tf.GradientTape() as dv_2:
      loss_1 = NT_G1.mse_loss_gnt(df) #
      h_a = tf.linalg.trace(tf.linalg.expm(tf.math.multiply(NT_G1.ADJ.weights[0], NT_G1.ADJ.weights[0])))-lat_dim_
      total_loss = loss_1 + alpha*h_a+rho*0.5*tf.math.abs(h_a)**2 +lamb*tf.norm(NT_G1.ADJ.weights[0], ord=1, axis=[-2,-1])
    loss_ms.append(loss_1)
    loss_t.append(total_loss)
    grad_1 = dv_1.gradient(total_loss, NT_G1.ADJ.trainable_variables)
    grad_2 = dv_2.gradient(total_loss, NT_G1.ENCODER.trainable_variables)
    basic_opt.apply_gradients(zip(grad_1, NT_G1.ADJ.trainable_variables))
    basic_opt2.apply_gradients(zip(grad_2, NT_G1.ENCODER.trainable_variables))

    h_a_new = tf.linalg.trace(tf.linalg.expm(tf.math.multiply(NT_G1.ADJ.weights[0], NT_G1.ADJ.weights[0])))-lat_dim_
    alpha =  alpha + rho * h_a_new
    if (tf.math.abs(h_a_new) >= gamma*tf.math.abs(h_a)):
        rho = beta*rho
    else:
        rho = rho
    if (i+1) %10 == 0: print(i+1, loss_1,total_loss)
    i = i+1

sns.set_style("darkgrid")
plt.plot(loss_t,color="darkred",label="total loss")
plt.legend()

sns.set_style("darkgrid")
plt.plot(loss_ms,color="darkred",label="z loss")
plt.legend()

sns.heatmap(np.array(NT_G1.ADJ.weights[0]), cmap="vlag",center=0)
plt.show()

np.mean(np.abs(np.array(NT_G1.ADJ.weights[0])))

#Estimated Graph
sns.heatmap(np.where(np.abs(np.array(NT_G1.ADJ.weights[0]))>np.quantile(np.abs(np.array(NT_G1.ADJ.weights[0])),0.75),1,0), cmap="gray_r",linewidths=1,linecolor="black")
plt.show()

#True Graph
sns.heatmap(true_dag,cmap="gray_r",linewidths=1,linecolor="black")


#(Hill-climbing algorithm based search)
model_hc = bn.structure_learning.fit(df, methodtype='hc')

model_hc['adjmat']*1

sns.heatmap(model_hc['adjmat']*1,cmap="gray_r",linewidths=1,linecolor="black")

