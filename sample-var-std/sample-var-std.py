import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    x=np.asarray(x)
    mean = x.sum()/x.size
    x_bar = np.full(x.size, mean)
    var = np.sum(np.square(x-x_bar))/(x.size-1)
    return(var,np.sqrt(var))