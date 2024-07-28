import torch
from config import Config
from model import Transformer
from train import train_step, evaluate
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import time

# Load dataset
dataset = load_dataset("wmt14", "de-en")


config = Config()

# Print all attributes and their values
for attr, value in config.__dict__.items():
    print(f"{attr}: {value}")

# Tokenizers
src_tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
tgt_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

Config.src_START_IDX = src_tokenizer.cls_token_id
Config.src_END_IDX = src_tokenizer.sep_token_id

Config.src_PAD_IDX = src_tokenizer.pad_token_id
Config.tgt_PAD_IDX = tgt_tokenizer.pad_token_id

def tensor_transform(token_ids, tokenizer, max_length):
    token_ids = token_ids[:max_length - 2]  # Reserve space for [CLS] and [SEP]
    return torch.tensor([tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id])

def collate_fn(batch, src_pad_idx, tgt_pad_idx, max_length):
    src_batch, tgt_batch = [], []
    for sample in batch:
        src_sample = sample['translation']['de']
        tgt_sample = sample['translation']['en']
        src_batch.append(tensor_transform(src_tokenizer.encode(src_sample), src_tokenizer, max_length))
        tgt_batch.append(tensor_transform(tgt_tokenizer.encode(tgt_sample), tgt_tokenizer, max_length))

    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)
    

    return src_batch, tgt_batch

def main():
    max_length = Config.max_seq_length

    train_size = len(dataset['train'])
    val_size = len(dataset['validation'])

    # Dataloaders
    train_dataloader = DataLoader(dataset['train'].select(range(min(Config.train_split, train_size))), batch_size=Config.batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, Config.src_PAD_IDX, Config.tgt_PAD_IDX, max_length))
    val_dataloader = DataLoader(dataset['validation'].select(range(min(Config.test_split, val_size))), batch_size=Config.batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, Config.src_PAD_IDX, Config.tgt_PAD_IDX, max_length))

    # Initialize model
    model = Transformer(Config).to(Config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)

    # Training and evaluation
    for epoch in range(Config.num_epochs):
        start_time = time.time()
        train_loss = 0
        for batch, (src, tgt) in enumerate(train_dataloader):
            src, tgt = src.to(Config.device), tgt.to(Config.device)
            # print(f"src shape: {src.shape}, tgt shape: {tgt.shape}")
            loss = train_step(src, tgt, model, optimizer, Config)
            train_loss += loss

           # if batch % 100 == 0:  # Print loss for every batch for quick debugging
            #    print(f'Epoch {epoch+1} Batch {batch} Loss {loss:.4f}')

        val_loss, bleu_score = evaluate(model, val_dataloader, src_tokenizer, tgt_tokenizer, Config)
        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch+1} Train Loss {train_loss/len(train_dataloader):.4f} Val Loss {val_loss:.4f} BLEU Score {bleu_score:.2f} Time {elapsed_time:.2f}s')

if __name__ == "__main__":
    main()
