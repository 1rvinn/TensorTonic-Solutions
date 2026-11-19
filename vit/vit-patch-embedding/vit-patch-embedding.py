import numpy as np

def patch_embed(image: np.ndarray, patch_size: int,
                W_proj: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Returns float64 patch embeddings with shape (B, N, D).
    """
    D = bias.shape[0]
    B, H, W, C = image.shape[0], image.shape[1], image.shape[2], image.shape[3]
    N = (H//patch_size) * (W//patch_size)
    Nh = (H//patch_size) 
    Nw = (W//patch_size)
    image_usable = image[:,:(H//patch_size)*patch_size, :(W//patch_size)*patch_size, :]
    image_alt = image_usable.reshape(B, Nh, patch_size, Nw, patch_size, C)
    image_alt = image_alt.transpose(0,1,3,2,4,5)
    image_flat = image_alt.reshape(B, N, patch_size*patch_size*C)
    image_proj = image_flat@W_proj + bias
    return image_proj