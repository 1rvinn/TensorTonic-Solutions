import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """
    return np.tanh(np.matmul(x_t, W_xh.transpose())+np.matmul(h_prev, W_hh.transpose())+b_h)