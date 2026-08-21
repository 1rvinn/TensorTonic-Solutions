import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    mask_positions: np.ndarray,
    replace_probs: np.ndarray,
    random_tokens: np.ndarray,
    mask_token_id: int = 103
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: tuple of (np.ndarray masked_ids, np.ndarray labels) with masking applied
    """
    masked_ids=np.zeros(token_ids.shape)
    labels=np.zeros(token_ids.shape)
    for i in range(token_ids.shape[0]):
        for j in range(token_ids.shape[-1]):
            if mask_positions[i][j]==True:
                if replace_probs[i][j]<0.8:
                    masked_ids[i][j]=mask_token_id
                elif replace_probs[i][j]>=0.9:
                    masked_ids[i][j]=token_ids[i][j]
                else:
                    masked_ids[i][j]=random_tokens[i][j]
                labels[i][j]=token_ids[i][j]
            else:
                masked_ids[i][j]=token_ids[i][j]
                labels[i][j]=-100
    return masked_ids, labels

class MLMHead:
    """Masked LM prediction head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token logits: hidden_states @ W + b
        """
        return np.matmul(hidden_states,self.W)+self.b