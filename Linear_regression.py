import numpy as np 
import matplotlib.pyplot as plt 
  
def linear_regression(x, y): 
    """ this calculates the regression coefficients for a given x and y""" 
    n = np.size(x) # number of values
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) # means of both
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x # 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx # there is an analytic answer so no need for any minimisation functions
    b_0 = m_y - b_1*m_x # with 15 mins effort you can rederive this
  
    return(b_0, b_1) # tuple bc doesn't change
  
def plot_regression_line(x, y, b): 
    """ plotting the actual points as scatter plot """
    plt.scatter(x, y, s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    # function to show plot 
    plt.show() 

# example

x = np.linspace(0,10,100)+np.random.randn(100)
y = 2*x + 3 + np.random.randn(100)
b = linear_regression(x,y)
plot_regression_line(x,y,b)