import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))


def lstm_forward(X: np.ndarray, W_f: np.ndarray, W_i: np.ndarray,
                 W_c: np.ndarray, W_o: np.ndarray, b_f: np.ndarray,
                 b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray,
                 W_y: np.ndarray, b_y: np.ndarray) -> dict:
    """
    Returns outputs, final_hidden_state, and final_cell_state.
    """
    N, T, input_dim=X.shape
    hidden_dim, _ = W_o.shape
    h_t = np.full((N,hidden_dim),0)
    c_t = np.full((N,hidden_dim),0)
    y=[] 

    for t in range(T):
        x_t = X[:,t,:]
        concat = np.concatenate((h_t,x_t),axis=-1)
        f_t = sigmoid(concat@W_f.T+b_f)
        i_t = sigmoid(concat@W_i.T+b_i)
        c_tilda_t = np.tanh(concat@W_c.T+b_c)
        c_t = c_t*f_t+i_t*c_tilda_t
        o_t = sigmoid(concat@W_o.T+b_o)
        o_tilda_t =  np.tanh(c_t)
        h_t = o_t*o_tilda_t
        y_t = h_t@W_y.T + b_y
        y.append(y_t) # y.shape = (T,N,output_dim)

    y=np.stack(y,axis=1) # stack wrt T which is the first axis
    
    return {"outputs":y.astype(np.float64),"final_hidden_state":h_t.astype(np.float64),"final_cell_state":c_t.astype(np.float64)}