from dataclasses import dataclass
import torch

@dataclass
class Config:
    src_vocab_size: int = 30000  # Size of the source vocabulary
    tgt_vocab_size: int = 28996  # Size of the target vocabulary
    n_layer: int = 6  # Number of transformer layers 
    n_head: int = 8  # Number of attention heads
    n_embd: int = 512  # Embedding size
    dropout: float = 0.2  # Dropout rate
    bias: bool = True
    batch_size: int = 64  # 32,48,64 Number of sequences processed in parallel
    learning_rate: float = 3e-4 
    num_epochs: int = 100
    max_seq_length: int = 128  # 64,128,256,512 Maximum sequence length
    #device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device: str = 'cuda'
    src_PAD_IDX: int = 0  # Source padding index, set by main
    tgt_PAD_IDX: int = 0  # Target padding index, set by main
    train_split: int = 450875 #total is 450875
    test_split: int = 3000 #total is 3000
    src_START_IDX: int = 101  # Source start token index
    src_END_IDX: int = 102  # Source end token index
 