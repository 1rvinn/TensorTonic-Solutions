import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    qkt=torch.matmul(Q,K.transpose(-2,-1))
    d_m = K.shape[-1]
    attention_weights=torch.softmax(qkt/math.sqrt(d_m), -1) # attention calc row wise, ie each row should sum to 1, ie softmax should happen across columns
    # dim = -2 does it column wise
    output = torch.matmul(attention_weights, V)
    return output