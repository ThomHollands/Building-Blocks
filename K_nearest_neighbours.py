# implementing KNN
import numpy as np
import matplotlib.pyplot as plt

def standardiser(X):
    """ Z-standardises our data"""
    rows,cols = X.shape
    X_std = np.zeros(X.shape)
    for row in rows:
        mean = np.mean(X[row][:])
        std = np.std(X[row][:])
        X_std[row][:] = (X[row][:]-mean)/std
    return X_std

def euclidean_distance(row1,row2):
    distance = 0.0
    for i in range(0,len(row1)): # (-1 bc we don't want to use the final column which is the output column)
        distance = distance + (row1[i]-row2[i])**2
    return np.sqrt(distance)

def get_nearest_neighbours(X_train,y_train,X_test_row,num_neighbours):
    distances = [] # initialise 
    for i in range(0,len(y_train)):
        dist = euclidean_distance(X_test_row, X_train[i][:]) # calc distance between each observation
        distances.append((X_train[i][:],y_train[i], dist)) # add 3 element tuple 
    distances.sort(key=lambda tup: tup[2]) # sort based on distance
    neighbours = []
    for i in range(0,num_neighbours):
        neighbours.append(distances[i][0:-1]) # adds the nearest neighbour data points and outputs
    return neighbours

def predict_class(X_train,y_train,X_test_row,num_neighbours):
    neighbours = get_nearest_neighbours(X_train,y_train,X_test_row,num_neighbours)
    output_values = [row[1] for row in neighbours] # these are the classes
    prediction = max(set(output_values), key=output_values.count)
    return prediction

def k_n_n(X_train,y_train,X_test,y_test,num_neighbours=10):
    predictions = []
    for row in X_test:
        output = predict_class(X_train,y_train,row,num_neighbours)
        predictions.append(output)
    return predictions

preds = k_n_n(X_train,list(y_train),X_test,list(y_test),num_neighbours=23)

def confusion_matrix(preds,y_test):
    # initialise confusion matrix
    # put preds on y, test on x
    c_matrix = [[0,0],
                [0,0]]
    for i in range(0,len(y_test)):
        if preds[i] == y_test[i]:
            if preds[i] == 1: # true positive
                c_matrix[0][0] = c_matrix[0][0]+1
            if preds[i] == 0: # true negative
                c_matrix[1][1] = c_matrix[1][1]+1
        else:
            if preds[i] == 1 and y_test[i] == 0: # false positive
                c_matrix[1][0] = c_matrix[1][0] + 1
            if preds[i] == 0 and y_test[i] == 1:
                c_matrix[0][1] = c_matrix[0][1] + 1
    return c_matrix
    
confusion_matrix(preds,list(y_test))