import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    T = X.shape[1]
    batch = X.shape[0]
    hidden_dim = h_0.shape[1]
    
    hidden_states = np.zeros((batch,T,hidden_dim))
    hidden_states[:,-1,:]=h_0
    for t in range(0,T):
        hidden_states[:,t,:]=np.tanh(np.matmul(X[:,t,:], W_xh.transpose()) + np.matmul(hidden_states[:,t-1,:], W_hh.transpose())+b_h)
    return (hidden_states, hidden_states[:,-1,:])