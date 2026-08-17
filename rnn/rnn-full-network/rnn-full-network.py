import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        batch=X.shape[0]
        T=X.shape[-2]
        
        if h_0 is None:
            h_0=np.zeros((batch, self.hidden_dim))
        
        input_dim=X.shape[-1]
        hidden_dim=h_0.shape[-1]
        output_dim=self.W_hy.shape[-2]
        
        y_seq=np.zeros((batch, T, output_dim))
        h_prev=h_0
        for i in range(T):
            X_scaled=np.matmul(X[:,i,:],self.W_xh.T)
            h_prev_scaled=np.matmul(h_prev,self.W_hh.T)
            h_current = np.tanh(X_scaled+h_prev_scaled+self.b_h)
            y_seq[:,i,:]=np.matmul(h_current,self.W_hy.T)+self.b_y
            h_prev=h_current
        h_final=h_prev
        return (y_seq, h_final)