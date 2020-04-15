# multivariable linear regression implementation
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
%matplotlib inline


def normaliser(X):
    ''' Z-normalises the array X'''
    if len(X.shape)>1:
        rows,n = X.shape
        X_norm = np.zeros(X.shape)
        for n_row in range(0,rows):
            mean = np.mean(X[n_row,:])
            std_dev = np.std(X[n_row,:])
            X_norm[n_row,:] = (X[n_row,:]-mean)/std_dev
    else:
        X_norm = np.zeros(len(X))
        mean = np.mean(X)
        std_dev = np.std(X)
        X_norm = (X-mean)/std_dev
    return X_norm


def cost_function(X, Y, B):
    ''' Defines a mean squared error cost function for X, Y and coefficients B'''
    m = len(Y) # length of observations
    J = np.sum((X.T.dot(B) - Y) ** 2)/(2 * m) # mean squared error
    return J


def linear_regression_gradient_descent(X, Y, B, alpha=0.005, iterations=2000):
    '''X is the independent variables. Y is the predicted dependent variables. B is the vector of regression
    coefficients. Alpha is the learning rate. iterations is the number of runs'''
    cost_history = [0] * iterations # number of runs
    m = len(Y) # length of output vector
 
    for iteration in range(iterations):
 # Hypothesis Values
        h = X.T.dot(B) # multiply B into X (returns vector)
 # Difference b/w Hypothesis and Actual Y
        loss = h - Y 
 # Gradient Calculation -- Adds up all the differences apportioned to their corresponding X value.
        gradient = X.dot(loss) / m # .T transposes X
 # Changing Values of B using Gradient
        B = B - alpha * gradient
 # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
 
    return B, cost_history

def predict(X,B):
    ''' for a given X and B, outputs the corresponding Y value'''
    y_pred = X.T.dot(B)
    return y_pred

# create data
data = 10*np.ones((5,500))
n = data.shape[0]
for i in range(0,len(data)-1):
    data[i,:]=np.linspace(0,100,num=500)
data = data + np.random.randn(n,500) # add noise
#data[-1,:] = 200*data[0,:]+400*data[1,:]
data[-1,:] = 30*data[0,:] -20*data[1,:] + 50*data[2,:] + -40*data[3,:] # make fake data
X = data[0:-1,:] # select independent variables
Y = data[-1,:] # select dependent variables
plt.plot(X.T,Y)
#normalise data
#X = normaliser(X)
# Split up test and training data
m = int(np.floor(0.9*len(Y)))
X_train = X[:,:m]
#X_train = np.c_[np.ones(len(X_train),dtype='int64'),X_train]
y_train = Y[:m] # first 90% of values
X_test = X[:,m:] # last 10% of values
#X_test = np.c_[np.ones(len(X_test),dtype='int64'),X_test]
y_test = Y[m:]

# initialise B
B_init = np.zeros(X_train.shape[0])
B_test = [30,-20,50,-40]

alpha = 0.0005
iter_ = 10000

newB, cost_history = linear_regression_gradient_descent(X_train, y_train, B_init, alpha, iter_)

y_pred = predict(X_train,newB)

ax1 = plt.subplot(121)
ax1.hist((y_train-y_pred),bins=20)
ax2 = plt.subplot(122)
ax2.loglog(cost_history)
print(newB)