# implement k means algorithm
import numpy as np
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=200, n_features=2, 
                           centers=2, cluster_std=1.8,random_state=101)
in_data = data[0]
def calc_centroid(data):
    totals = np.mean(data,axis=0)
    return totals

def calc_distance(x1,x2):
    squares = np.square(x1-x2)
    sum_squares = np.sum(squares)
    distance = np.sqrt(sum_squares)
    return distance

def k_means(data,k=2):
    change_val = 1
    # create clusters
    k_vals = np.zeros(data.shape[0]) # initialise
    centroid_locs = np.zeros((k,data.shape[1]))
    # create initial k probabilities
    k_prob = np.array(range(0,k)) / float(k)    
    # randomly assign data to cluster
    for item_ind,item in enumerate(data):
        delta_old = 1.0
        u = np.random.uniform(low=0.0,high=1.0)
        for i,prob in enumerate(k_prob): # check which cluster to put this row in
            delta_new = abs(u-prob)
            if delta_new < delta_old:
                delta_old = delta_new
                k_ind = i
                
        # add to correct cluster        
        k_vals[item_ind] = k_ind
    
    #while change_val > 0.1:
    counter = 0
    while counter < 100:
        change_val = 0
    # calc the centroid of each cluster
        for k_val in range(0,k):
            # get indices of data in each k_val
            # calculate the centroid for that
            cen_data = []
            for ind,item in enumerate(k_vals):
                if k_val == item:
                    cen_data.append(data[ind])
            centroid = calc_centroid(cen_data)
            centroid_locs[k_val] = centroid
            
    # for each data point, check which centroid is closest
        for item_ind,item in enumerate(data):
            dist_old = 10e100
            for c_ind,centroid in enumerate(centroid_locs):
                dist_new = calc_distance(item,centroid)
                if dist_new < dist_old: # change the centroids
                    dist_old = dist_new
                    change_val = change_val + abs(k_vals[item_ind] - c_ind)
                    #print(abs(k_vals[item_ind]-c_ind))
                    
                    #print(change_val)
                    k_vals[item_ind] = c_ind
        counter = counter + 1
    return data, k_vals
    # reallocate that data point to the new centroid - append to new cluster, delete from old cluster
    # if points change, store value
    # if no points change, set change_val to 0
    
    # repeat
out_data,preds = k_means(in_data,k=2)
sum(abs(preds-data[1]))