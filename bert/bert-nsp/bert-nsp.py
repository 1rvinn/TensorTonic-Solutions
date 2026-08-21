import numpy as np
from typing import List, Tuple

def create_nsp_pairs(
    documents: List[List[str]],
    pair_specs: List[dict]
) -> List[Tuple[str, str, int]]:
    """
    Returns: list of (sentence_A, sentence_B, is_next_label) tuples
    """
    res = []
    for spec in pair_specs:
        sentenceA=documents[spec["doc_a"]][spec["sent_a"]]
        sentenceB=documents[spec["doc_b"]][spec["sent_b"]]
        if spec["doc_a"]==spec["doc_b"] and spec["sent_b"]==spec["sent_a"]+1:
            is_next_label=1
        else:
            is_next_label=0
        res.append((sentenceA, sentenceB, is_next_label))
    return res
    
class NSPHead:
    """Next Sentence Prediction classification head."""
    
    def __init__(self, hidden_size: int):
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)
    
    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        """
        Predict IsNext logits: cls_hidden @ W + b
        """
        return np.matmul(cls_hidden, self.W)+self.b

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax along last axis."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
