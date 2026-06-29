import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    positional_encodings=[]
    for pos in range(seq_length):
        encoding=[]
        for i in range(d_model):
            if i%2==0:
                encoding.append(np.sin(pos/(10000**(i/d_model))))
            else:
                encoding.append(np.cos(pos/(10000**((i-1)/d_model))))
        print(encoding)
        positional_encodings.append(encoding)
    print(positional_encodings)
    return np.array(positional_encodings)