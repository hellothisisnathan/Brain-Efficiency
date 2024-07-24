import numpy as np
import math
import random
from matplotlib import pyplot as plt
from IPython.display import clear_output

x = np.linspace(0, 100, 2000)
y = (1/7)*np.sqrt(2500+np.power(x,2))+(1/2)*np.sqrt(2500+np.power((100-x),2))

def get_rand_num(min_val, max_val):
    range = max_val - min_val
    choice = random.uniform(0,1)
    return min_val + range * choice

def f_of_x(x):
    """
    This is the main function we want to integrate over.
    Args:
    - x (float) : input to function; must be in radians
    Return:
    - output of function f(x) (float)
    """
    return (1/7)*np.sqrt(2500+np.power(x,2))+(1/2)*np.sqrt(2500+np.power((100-x),2))

def crude_monte_carlo(num_samples=5000):
    lower_bound = 0
    upper_bound = 100

    min = np.inf
    s = 0
    for i in range(num_samples):
        x = get_rand_num(lower_bound, upper_bound)
        f = f_of_x(x)
        if min > f:
            min = f
            s = x

    return min, s
f,s = crude_monte_carlo(10000)
print(s, f)
plt.plot(x, y)
plt.plot(s,f, 'ro')
plt.show()