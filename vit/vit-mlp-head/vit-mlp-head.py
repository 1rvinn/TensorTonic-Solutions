import numpy as np

def classification_head(encoder_output: np.ndarray,
                        W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C).
    """
    cls = encoder_output[:,0,:]
    mu = np.mean(cls, axis = -1, keepdims = True)
    variance = np.var(cls, axis = -1, keepdims = True)
    normalised  = (cls - mu)/np.sqrt(variance+1e-6)
    logits = normalised@W_head
    
    return logits