# coding: utf-8

# In[94]:

#get_ipython().magic('matplotlib inline')
import numpy as np 
import os
import sys
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU


# In[95]:

# fix random seed for reproducibility
np.random.seed(7)
V=np.random.randn(10)
print(V.dot(V), np.linalg.norm(V)**2)


# In[121]:

d=10 ##number of measurements per patient per time unit
k=2 ##complexity of the time series
n=200 ## number of patients
train_split = 50 ## number of patients training
len_sequence = 1

# stateful rnn has tsteps = 1
tsteps = 1
epochs = 25

T=200 ## amount of time measured for each patient (also batch size)
sigmaw=0.2 ##randomness
#d by d
U=np.random.randn(d,k).dot(np.random.randn(k,d))
betashift=np.random.randn(d)/np.sqrt(d)  ## used to check when we switch to the other time series
#normalized U
A=0.9*U/np.linalg.norm(U,ord=2)
#V is also d by d
V=np.random.randn(d,k).dot(np.random.randn(k,d))
#normalized V
B=0.9*V/np.linalg.norm(V,ord=2)
X = np.zeros((n,T,d)) #this represents n different time series, in d-dimensions, for T time-steps
Y = np.zeros((n,T - len_sequence,d)) #this is just X displaced by 1
flag=1


# In[122]:

print(X.shape)
print(Y.shape)


# In[123]:

for j in range(n):
    X[j,0,:]=np.random.randn(d)/np.sqrt(d)
    flag=1
    for i in range(1,T):      
        if flag and ((X[j,i-1,:].dot(betashift))>.5): ##if flag was true and we go above a certain threshold, then set flag=0
            flag=0
        ## we can think of flag has flag==1 means not critical, flag==0 means critical
        if flag:
            X[j,i,:]=A.dot(X[j,i-1,:])+sigmaw*np.random.randn(d)
        else:
            X[j,i,:]=B.dot(X[j,i-1,:])+sigmaw*np.random.randn(d)
    for i in range(0, T - len_sequence):
        Y[j, i, :] = X[j, i+len_sequence, :]
X = X[:, 0:T-len_sequence, :]
## X is a n by d by T tensor
## Y is just X but shifted one backwards
## n represents independent trials (think of them as patients or time periods for patients separated)
## T represents the time horizon (how long we observe a patient, for now T is the same for everybody, but we can imagine that changing)
## d is the number of measurements that we have per patient per time.


# In[124]:

#Confirm that the sliding works
print(X[0, len_sequence, :])
print(Y[0,0,:])


# In[125]:

crazy = np.array([X[:, i: i + 20] for i in range(X.shape[1] - 21)])
print(crazy.shape)
crazy_y = np.array([X[:, i+21] for i in range(X.shape[1] - 21)])
print(crazy_y.shape)


# In[ ]:

#First model: Using the sliding window
model = Sequential()
model.add(GRU(128,activation='relu', input_shape=(crazy.shape[2:])))
model.add(Dense(d))
model.compile(loss='mean_squared_error', optimizer='adam')

for i in range(train_split): # for each patient
    print ("patient "+str(i))
    this_patient_x = crazy[:,i,:,:]
    this_patient_y = crazy_y[:,i,:]
    for ii in range(crazy.shape[0]): #for each window of time sequence
        this_patient_x_i = np.array([this_patient_x[ii,:,:]])
        this_patient_y_i = np.array([this_patient_y[ii,:]])
        model.fit(this_patient_x_i, this_patient_y_i, nb_epoch=1, batch_size=1, verbose=2)
    model.reset_states()


# In[ ]:

#Check evaluation of the model

