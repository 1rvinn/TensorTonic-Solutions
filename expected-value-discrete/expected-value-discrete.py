import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x=np.asarray(x)
    p=np.asarray(p)
    if np.any(p<0):
        raise ValueError('probs must be >0')
    if np.sum(p)!=1:
        raise ValueError('sum of probs must be 1')
    if x.shape!=p.shape:
        raise ValueError("shape mismatch")
    return np.dot(x,p)