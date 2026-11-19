import numpy as np

def prepend_class_token(patches: np.ndarray,
                        cls_token: np.ndarray) -> np.ndarray:
    """
    Returns the float64 sequence with the class token at position zero.
    """
    B = patches.shape[0]
    cls_token = np.repeat(cls_token, B, axis = 0)
    patches_ext = np.concatenate((cls_token, patches), axis=-2)
    print(patches.shape)
    print(patches_ext.shape)
    return patches_ext