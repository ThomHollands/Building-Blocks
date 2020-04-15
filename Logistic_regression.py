# implementing logistic regression
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

class LogRegression:
    def __init__(self,learning_rate=0.01,num_iter=100000,fit_intercept=False,verbose=True):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        
    def __add_intercept(self,X):
        intercept = np.ones((X.shape[0], 1)) # creates an intercept parameter
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self,z):
        """ computes a sigmoid function of input. Can handle arrays thanks to numpy"""
        return 1 / (1 + np.exp(-z))

    def __entropy_loss(self,h, y):
        """h is the probability of being in class 1. y is the true value. This is a mutual entropy loss function.
        It is minimised (unsurprisingly) when y = h"""
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() # this is the same as summing all then dividing by N

    def __calc_gradients(self,X,h,y): # calculates the gradient
        gradient = np.dot(X.T, (h - y)) / y.size
        return gradient

    def __update_weights(self,weights,gradient,learning_rate):
        new_weights = weights - learning_rate*gradient # updates the weights
        return new_weights
    
    def fit(self,X,y):
        """ trains the model"""
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
         # weights initialization
        self.weights = np.zeros(X.shape[1])
        
        for i in range(self.num_iter): # start iteration loop
            z = np.dot(X, self.weights) # linear regression dot product
            h = self.__sigmoid(z) # put into sigmoid
            gradient = self.__calc_gradients(X,h,y)
            self.weights = self.__update_weights(self.weights,gradient,self.learning_rate)
            
            if(self.verbose == True and i % 10000 == 0): # print stuff every 10k runs
                z = np.dot(X, self.weights)
                h = self.__sigmoid(z)
                print(f'loss: {self.__entropy_loss(h, y)} \t')
                
    def __predict_probs(self,X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X,self.weights))

    def predict(self,X,threshold=0.5):
        return self.__predict_probs(X) >= threshold

# load in iris data set        
iris = sklearn.datasets.load_iris() 
X = iris.data[:, :2]
y = (iris.target != 0) * 1
model = LogRegression(learning_rate=0.01, num_iter=100000)
%time model.fit(X, y)

preds = model.predict(X)
# checking for accuracy
print((preds == y).mean())
print(preds-y)