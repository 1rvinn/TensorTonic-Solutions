import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    s=np.linalg.norm(W_hh,ord=2)
    norms=[]
    for i in range(T):
        norms.append(s**i)
    return norms