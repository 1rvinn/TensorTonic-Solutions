import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # normalise
    mu = x.mean(axis=-1,keepdims=True)
    var = x.var(axis=-1,keepdims=True)
    z = (x-mu)/np.sqrt(var+eps)
    # scale and shift
    result = gamma*z+beta
    return result

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    bs = Q.shape[0]
    sl = Q.shape[1]
    dm = Q.shape[2]

    assert dm%num_heads==0, 'dm must be divisible by num_heads'    
    dk=dm//num_heads
    
    Q_hat=np.matmul(Q, W_q)
    K_hat=np.matmul(K, W_k)
    V_hat=np.matmul(V, W_v)

    Q_hat = Q_hat.reshape(bs, sl, num_heads, dk).transpose(0,2,1,3)
    K_hat = K_hat.reshape(bs, sl, num_heads, dk).transpose(0,2,1,3)
    V_hat = V_hat.reshape(bs, sl, num_heads, dk).transpose(0,2,1,3)

    attention=np.matmul(softmax(np.matmul(Q_hat, K_hat.transpose(0,1,3,2))/np.sqrt(dk)), V_hat)

    attention=attention.transpose(0,2,1,3).reshape(bs, sl, dm)
    
    result = np.matmul(attention, W_o)
    
    return result

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    return np.matmul(np.maximum(np.matmul(x,W1)+b1,0), W2)+b2
    pass

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    x_dash = layer_norm(multi_head_attention(x,x,x,W_q,W_k,W_v,W_o,num_heads)+x, gamma1, beta1)
    output = layer_norm(feed_forward(x_dash, W1, b1, W2, b2)+x_dash, gamma2, beta2)
    return output