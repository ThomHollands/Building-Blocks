import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

data = np.random.randn(n)


def KDE(data_array, bandwidth=None):
    
    """ Plots the Kernel Density Estimate for a 1D array of data, showing the intermediate 
    step of plotting the individual kernels.
    """
    data = data_array
    d_range = data.max()-data.min()
    n = len(data)
    x_axis = np.linspace(data.min()-0.4*d_range,data.max()+0.4*d_range,100)

    sigma = np.std(data)
    if bandwidth is None:
        bandwidth = ((4*(sigma**5))/(3*n))**(0.2) # std dev of kernel distributions
        
    kernel_list = []
    for point in data:
    # normal distribution at each point
        kernel = stats.norm(point,bandwidth).pdf(x_axis)
        kernel_list.append(kernel)
        
    #Scale for plotting
        kernel = kernel / kernel.max()
        kernel = kernel *0.25
        plt.plot(x_axis,kernel,color = 'grey',alpha=0.5)
    #plt.ylim(0,0.6)

    sum_of_kernels = np.sum(kernel_list,axis=0)*(1/n) # total kernels
    
    plt.plot(x_axis,sum_of_kernels,color='indianred')
    sns.rugplot(data,c='indianred')
    plt.title('sum of the basis functions')

KDE(data)