import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    batch=X.shape[0]
    T=X.shape[1]
    hidden_dim=h_0.shape[-1]
    hidden_states = np.zeros((batch, T, hidden_dim))
    h_prev=h_0
    for i in range(T):
        hidden_states[:,i,:]=np.tanh(np.matmul(X[:,i,:],W_xh.T)+np.matmul(h_prev,W_hh.T)+b_h)
        h_prev=hidden_states[:,i,:]
    return (hidden_states, h_prev)