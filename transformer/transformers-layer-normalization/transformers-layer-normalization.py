import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    mu = x.mean(axis=-1, keepdims=True) # need to keep dimensions so that subtraction happens fine
    sigma_sqrd = x.var(axis=-1, keepdims=True) 
    # center
    z = (x-mu)/np.sqrt(sigma_sqrd+eps)
    # scale and shift
    result = gamma*z + beta
    return result