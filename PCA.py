# implementing PCA from scratch

import numpy as np

np.random.seed(2247) # random seed for consistency

# generate two sets of random data
mu_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T

mu_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T


%matplotlib inline
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
# visualise the data in 3d
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10   
ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=8, alpha=0.5, color='red', label='class2')

plt.title('Samples for class 1 and class 2')
ax.legend(loc='upper right')

plt.show()


# take the whole data set ignoring class labels

all_samples = np.concatenate((class1_sample, class2_sample), axis=1)

# compute the mean vector along each axis
def compute_mean_vec(data):
    n_rows = data.shape[0]
    mean_vec = np.zeros((n_rows,1))
    for i in range(0,n_rows):
        mean_vec[i,:] = np.mean(data[i,:])
    return mean_vec

mean_vector = compute_mean_vec(all_samples)

# calc scatter matrix -- an estimator for the covariance matrix

def calc_scatter_mat(data):
    n_rows,n_cols = data.shape
    scat_mat = np.zeros((n_rows,n_rows))
    mean_vector = compute_mean_vec(data)
    for i in range(n_cols):
        delta = (data[:,i].reshape(n_rows,1) - mean_vector).dot((data[:,i].reshape(n_rows,1) - mean_vector).T)
        #print(delta)
        scat_mat = scat_mat + delta
    return scat_mat

scatter_matrix = calc_scatter_mat(all_samples)


# eigenvectors and eigenvalues for the from the scatter matrix
def calc_eigenpairs(matrix):
    eig_vals,eig_vecs = np.linalg.eig(matrix)
    n = len(eig_vals)
    pairs = []
    for i in range(n):
        pairs.append([eig_vals[i],eig_vecs[i,:]])
    return pairs

eig_pairs = calc_eigenpairs(scatter_matrix)
print(eig_pairs)

# sort eigeinpairs from high to low
def sort_eigenpairs(pairs):
    sorted_pairs= sorted(pairs,key=lambda x:x[0],reverse=True)
    return sorted_pairs

eig_pairs = sort_eigenpairs(eig_pairs)

# choose k vectors with biggest eigenvalues
def reduce_dim_mat(eig_pairs,k=2):
    n_rows = len(eig_pairs)
    w_mat = np.zeros((n_rows,1,k))
    for i in range(k):
        w_mat[:,:,i] = eig_pairs[i][1].reshape(3,1)
    return np.squeeze(w_mat)

matrix_w = reduce_dim_mat(eig_pairs)

# transform the data by dotting it all in
transformed = matrix_w.T.dot(all_samples)

# plot the transformed data
plt.plot(transformed[0,0:20], transformed[1,0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(transformed[0,20:40], transformed[1,20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples with class labels')

plt.show()