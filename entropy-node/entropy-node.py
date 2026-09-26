import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    unique=set(y)
    freqs=[]
    for element in unique:
        count=0
        for i in y:
            if i==element:
                count+=1
        freqs.append(count)
    freqs=np.array(freqs)
    p=freqs/len(y)
    logp=np.log(p)/np.log(2)
    entropy=-np.dot(p,logp)
    return entropy