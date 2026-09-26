import numpy as np

def gini_impurity(y_left: list, y_right: list) -> float:
    """
    Returns the impurity as a float.
    """
    y_left=np.array(y_left)
    y_right=np.array(y_right)
    unqi_l, freq_l = np.unique(y_left, return_counts=True)
    unqi_r, freq_r = np.unique(y_right, return_counts=True)
    p_left = freq_l/len(y_left)
    p_right = freq_r/len(y_right)
    g_left=1
    for p in p_left:
        g_left-=p**2
    g_right=1
    for p in p_right:
        g_right-=p**2
    if len(y_left)==0 and len(y_right)==0:
        return 0
    g_split = (len(y_left)*g_left + len(y_right)*g_right)/(len(y_left)+len(y_right))
    return g_split