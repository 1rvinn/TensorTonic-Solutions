import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        
        special_tokens=[self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        # for id, word in enumerate(special_tokens):
        #     self.word_to_id[word]=id
        #     self.id_to_word[id]=word
        
        unique_words = set()

        for text in texts:
            unique_words|=set(text.lower().split())

        unique_words=sorted(unique_words)
        unique_words=special_tokens+unique_words
        
        for id, word in enumerate(unique_words):
            if word not in self.word_to_id:
                self.word_to_id[word]=id
                self.id_to_word[id]=word
                
        self.vocab_size = len(self.word_to_id)
        
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        encodings=[]
        
        for word in text.lower().split():
            encodings.append(self.word_to_id.get(word,1))

        return encodings
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """

        decodings = []

        for id in ids:
            decodings.append(self.id_to_word.get(id,self.unk_token))
        
        return " ".join(decodings)
