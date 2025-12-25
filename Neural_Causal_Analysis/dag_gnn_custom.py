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


#Custom DAG_GNN 정의

class Generalized_DAG_GNN(tf.keras.Model):
  def __init__(self, input_dim,lat_dim,h=[20,20]):
    super(Generalized_DAG_GNN,self).__init__()
    self.input_dim = input_dim
    self.lat_dim = lat_dim
    self.h1 = h[0]*lat_dim
    self.h2 = h[1]*lat_dim


    mat1 = np.zeros(self.input_dim*self.h1).reshape(self.input_dim,self.h1)
    mat2 = np.zeros(self.h1*self.h2).reshape(self.h1,self.h2)
    mat3 = np.zeros(self.h2*self.lat_dim).reshape(self.h2,self.lat_dim)
    mat4 = np.ones(self.lat_dim*self.lat_dim).reshape(self.lat_dim,self.lat_dim)
    for i in range(self.lat_dim):
      mat1[i,(i*h[0]):((i+1)*h[0])] = 1
    for i in range(self.lat_dim):
      mat2[(i*h[0]):((i+1)*h[0]),(i*h[1]):((i+1)*h[1])] = 1
    for i in range(self.lat_dim):
      mat3[(i*h[1]):((i+1)*h[1]),i] = 1
    for i in range(self.lat_dim):
      mat4[i,i] = 0

    class mask_1(tf.keras.constraints.Constraint):
      def __call__(self,w):
        return mat1*w

    class mask_2(tf.keras.constraints.Constraint):
      def __call__(self,w):
        return mat2*w

    class mask_3(tf.keras.constraints.Constraint):
      def __call__(self,w):
        return mat3*w

    class mask_4(tf.keras.constraints.Constraint):
      def __call__(self,w):
        return mat4*w


    self.ADJ = self.add_weight(shape=(self.lat_dim,self.lat_dim),trainable=True,name="adj",initializer="zeros",constraint=mask_4())

    self.ENCODER = Sequential([
        InputLayer(shape=(self.lat_dim,)),
        Dense(self.h1,activation="relu",kernel_initializer="glorot_normal",use_bias=False,kernel_constraint=mask_1()),
        Dense(self.h2,activation="relu",kernel_initializer="glorot_normal",use_bias=False,kernel_constraint=mask_2()),
        Dense(self.lat_dim,activation="linear",kernel_constraint=mask_3(),use_bias=False)
    ])

    self.DECODER = Sequential([
        InputLayer(shape=(self.lat_dim,)),
        Dense(self.h1,activation="relu",kernel_initializer="glorot_normal",use_bias=False,kernel_constraint=mask_1()),
        Dense(self.h2,activation="relu",kernel_initializer="glorot_normal",use_bias=False,kernel_constraint=mask_2()),
        Dense(self.lat_dim,activation="linear",kernel_constraint=mask_3(),use_bias=False)
    ])

    self.Enc_v1 = Sequential([
      InputLayer(shape=(self.lat_dim,)),
      Dense(self.lat_dim+self.lat_dim, activation="linear",use_bias=False)
    ])

  def reparam(self,x):
    ec = self.Enc_v1(x, training=True)
    mean, lv = tf.split(ec,num_or_size_splits=2,axis=1)
    eps = tf.random.normal(shape=mean.shape)
    return eps*tf.math.exp(lv*0.5) + mean


  def enc(self,x):
    a1 = self.ENCODER(x,training=True)
    I = tf.eye(self.lat_dim, dtype=self.ADJ.dtype)
    M = I-self.ADJ
    Z_1 = tf.linalg.matmul(a1,M)
    Z = self.reparam(Z_1)
    return Z


  def dec(self,x,sigmoid=False):
    x_T = tf.transpose(x)
    I = tf.eye(self.lat_dim, dtype=self.ADJ.dtype)
    M = I-self.ADJ
    a2_T = tf.linalg.solve(M, x_T)
    a2 = tf.transpose(a2_T)
    Z_2 = self.DECODER(a2, training=True)
    X_hat = self.reparam(Z_2)
    if sigmoid == True:
      X_hat = tf.math.sigmoid(X_hat)
    return X_hat

  def mse_loss_(md,x):
    z = md.enc(x)
    x_hat = md.dec(z,sigmoid=True)
    x_ = tf.convert_to_tensor(x)
    mse = tf.keras.losses.MeanSquaredError()
    loss_1 = mse(x_,x_hat)
    loss_2 = tf.norm(z, ord=2)
    loss = loss_1 + loss_2/(x.shape[0])
    return 0.5*loss

#Example
dgn = Generalized_DAG_GNN(5,5,[6,8])

dgn.DECODER.summary()

np.array([[1,2,4,2,1],[1,3,2,1,2]]).shape

dgn.dec(dgn.enc(np.array([[1,2,4,2,1],[1,3,2,1,2]])))

dgn.mse_loss_(np.array([[1,2,4,2,1],[1,3,2,1,2]]))

dgn.trainable_variables[0]


#Importing causal data
np.random.seed(321)
dg = bn.import_DAG("sprinkler")
df = bn.sampling(dg, n=200)


true_dag = bn.dag2adjmat(dg['model'])*1

#Ground Truth causal DAG
true_dag


#Training

tf.keras.utils.set_random_seed(678)


Epochs = 500
lat_dim_ = 4
G_D = Generalized_DAG_GNN(lat_dim_,lat_dim_,[6,6])

alpha=0.6
i = 0
rho = 0.1
gamma=0.9
beta = 1.01
lamb = 1.0 #L1-regularization


loss_ms = []
loss_t = []
basic_opt = tf.keras.optimizers.Adam(learning_rate=0.01)
basic_opt2 = tf.keras.optimizers.Adam(learning_rate=0.005)
basic_opt3 = tf.keras.optimizers.Adam(learning_rate=0.005)
mse = tf.keras.losses.MeanSquaredError()

while i < Epochs:
    with tf.GradientTape() as dv_1, tf.GradientTape() as dv_2, tf.GradientTape() as dv_3:
      loss_1 = G_D.mse_loss_(df)
      h_a = tf.linalg.trace(tf.linalg.expm(tf.math.multiply(G_D.weights[0], G_D.weights[0])))-lat_dim_
      total_loss = loss_1 + alpha*h_a+rho*0.5*tf.math.abs(h_a)**2+lamb*tf.norm(G_D.weights[0], ord=1, axis=[-2,-1])
    loss_ms.append(loss_1)
    loss_t.append(total_loss)
    grad_1 = dv_1.gradient(total_loss, [G_D.trainable_variables[0]])
    grad_2 = dv_2.gradient(total_loss, G_D.ENCODER.trainable_variables)
    grad_3 = dv_3.gradient(total_loss, G_D.DECODER.trainable_variables)
    basic_opt.apply_gradients(zip(grad_1, [G_D.trainable_variables[0]]))
    basic_opt2.apply_gradients(zip(grad_2, G_D.ENCODER.trainable_variables))
    basic_opt3.apply_gradients(zip(grad_3, G_D.DECODER.trainable_variables))

    h_a_new = tf.linalg.trace(tf.linalg.expm(tf.math.multiply(G_D.weights[0], G_D.weights[0])))-lat_dim_
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

sns.heatmap(np.array(G_D.weights[0]), cmap="vlag",center=0)
plt.show()

np.array(G_D.weights[0])

np.mean(np.abs(np.array(G_D.weights[0])))
np.quantile(np.abs(np.array(G_D.weights[0])),0.7)

#Binarized adjacency matrix
sns.heatmap(np.where(np.abs(np.array(G_D.weights[0]))>np.quantile(np.abs(np.array(G_D.weights[0])),0.75),1,0), cmap="gray_r",linewidths=1,linecolor="black")
plt.show()

sns.heatmap(true_dag, cmap="gray_r",linewidths=1, linecolor="black")


#Hill climbing algorithm과의 비교
model_hc = bn.structure_learning.fit(df, methodtype='hc',scoretype='bic')

model_hc['adjmat']*1

sns.heatmap(model_hc['adjmat']*1,cmap="gray_r",linewidths=1,linecolor="black")

sns.heatmap(true_dag,cmap="gray_r",linewidths=1,linecolor="black")


#Structural Hamming Distance(SHD)
shd1 = np.sum(np.sum(np.abs(model_hc['adjmat']*1-true_dag)));shd1

shd2 = np.sum(np.sum(np.abs(np.where(np.abs(np.array(G_D.weights[0]))>np.quantile(np.abs(np.array(G_D.weights[0])),0.75),1,0)-true_dag)));shd2

