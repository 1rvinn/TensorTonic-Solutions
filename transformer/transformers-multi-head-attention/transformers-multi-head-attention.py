import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batchsize = Q.shape[0]
    seqlen = Q.shape[1]
    
    # dimensions of model and dimensions of head
    d_m=Q.shape[-1]
    assert d_m%num_heads==0, "model dimension must be divisible by no of heads"
    d_k=d_m//num_heads
    
    # scaling Q,K,V
    Q_hat = np.matmul(Q,W_q)
    K_hat = np.matmul(K,W_k)
    V_hat = np.matmul(V,W_v)

    # dividing them headwise
    # (batchsize, seqlen, d_m) -> (batchsize, num_heads, seqlen, d_k)
    # for the above transformation, we must firstly resize (batchsize, seqlen, d_m) -> (batchsize, seqlen, num_heads, d_k)
    Q_hat = Q_hat.reshape(batchsize, seqlen, num_heads, d_k)
    K_hat = K_hat.reshape(batchsize, seqlen, num_heads, d_k)
    V_hat = V_hat.reshape(batchsize, seqlen, num_heads, d_k)
    # now we need to take transpose (batchsize, seqlen, num_heads, d_k) -> (batchsize, num_heads, seqlen, d_k)
    Q_hat = np.transpose(Q_hat,(0,2,1,3))
    K_hat = np.transpose(K_hat,(0,2,1,3))
    V_hat = np.transpose(V_hat,(0,2,1,3))

    # need to calc attentions
    attention=np.matmul(softmax(np.matmul(Q_hat,np.transpose(K_hat, (0,1,3,2)))/np.sqrt(d_k)),V_hat)
    
    # concatenate all
    # batchsize, numheads, seqlen, dk -> batchsize, seqlen, numheads, dk
    attention = np.transpose(attention,(0,2,1,3))
    # batchsize, seqlen, numheads, dk -> batchsize, seqlen, dm
    attention = attention.reshape(batchsize, seqlen, d_m)
    
    # scale by output weights
    result = np.matmul(attention,W_o)
    
    return result