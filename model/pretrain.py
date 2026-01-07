import torch
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse
import random
from tqdm import tqdm
import logging
from dataset import GenRecDataset

MASK_TOKEN_ID = 2049 

class AVATARPretrain(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(AVATARPretrain, self).__init__()
        
        t5config = T5Config(
            num_layers=config['num_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            vocab_size=config['vocab_size'],
            pad_token_id=config['pad_token_id'],
            eos_token_id=config['eos_token_id'],
            decoder_start_token_id=config['pad_token_id'],
            feed_forward_proj=config['feed_forward_proj'],
        )
        self.model = T5ForConditionalGeneration(t5config)
        
    @property
    def n_parameters(self):
      num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
      total_params = num_params(self.parameters())
      emb_params = num_params(self.model.get_input_embeddings().parameters())
      return (
          f'#Embedding parameters: {emb_params}\n'
          f'#Non-embedding parameters: {total_params - emb_params}\n'
          f'#Total trainable parameters: {total_params}\n'
      )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
      outputs = self.model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          labels=labels
      )
      return outputs.loss, outputs.logits

class PretrainDataset(GenRecDataset):
    """
    Pre-training dataset with Compound Masking Strategy (Random + Complement).
    """
    def __init__(self, mask_prob=0.2, text_mask_prob=0.3, image_mask_prob=0.3, **kwargs):
        """
        Args:
            mask_prob: Probability of Random Masking mode.
            text_mask_prob: Probability of Text-Only Masking mode (Image predicts Text).
            image_mask_prob: Probability of Image-Only Masking mode (Text predicts Image).
        """
        super().__init__(**kwargs)
        self.mask_prob = mask_prob

        self.strategy_probs = {
            'random': 0.4,
            'mask_text': 0.3,
            'mask_image': 0.3
        }
        
        self.num_tokens_per_item = 8 
        self.pretrain_max_len = self.max_len * self.num_tokens_per_item

    def _prepare_data(self):
        return super()._prepare_data()

    def __getitem__(self, index):
        item_data = self.data[index]
        
        full_seq = []
        history_text = item_data['history_text']
        history_image = item_data['history_image']
        
        min_len = min(len(history_text), len(history_image))
        
        for i in range(min_len):
            full_seq.extend(history_text[i])
            full_seq.extend(history_image[i])
        
        padding_len = self.pretrain_max_len - len(full_seq)
        if padding_len > 0:
            full_seq = [self.PAD_TOKEN] * padding_len + full_seq
        else:
            full_seq = full_seq[-self.pretrain_max_len:]
            
        corrupted_seq = []

        real_start_idx = 0
        for i, token in enumerate(full_seq):
            if token != self.PAD_TOKEN:
                real_start_idx = i
                break
        
        corrupted_seq.extend(full_seq[:real_start_idx])
        
        real_tokens = full_seq[real_start_idx:]
        num_real_items = len(real_tokens) // 8
        
        for i in range(num_real_items):
            chunk = real_tokens[i*8 : (i+1)*8]
            text_part = chunk[:4]
            image_part = chunk[4:]
            
            rand = random.random()
            if rand < self.strategy_probs['mask_text']:
                masked_chunk = [MASK_TOKEN_ID] * 4 + image_part
            elif rand < self.strategy_probs['mask_text'] + self.strategy_probs['mask_image']:
                masked_chunk = text_part + [MASK_TOKEN_ID] * 4
            else:
                masked_chunk = []
                for token in chunk:
                    if random.random() < self.mask_prob:
                        masked_chunk.append(MASK_TOKEN_ID)
                    else:
                        masked_chunk.append(token)
            
            corrupted_seq.extend(masked_chunk)

        return {
            "corrupted_input": torch.tensor(corrupted_seq, dtype=torch.long),
            "original_target": torch.tensor(full_seq, dtype=torch.long),
        }

def pretrain_collate_fn(batch, pad_token_id=0):
    corrupted_inputs = torch.stack([item['corrupted_input'] for item in batch])
    original_targets = torch.stack([item['original_target'] for item in batch])
    attention_mask = (corrupted_inputs != pad_token_id).long()
    
    return {
        'corrupted_input': corrupted_inputs,
        'original_target': original_targets,
        'attention_mask': attention_mask
    }

def pretrain_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc="Pre-training"):
        input_ids = batch['corrupted_input'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['original_target'].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVATAR Pre-training configuration")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--d_kv', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--vocab_size', type=int, default=2050)
    parser.add_argument('--pad_token_id', type=int, default=0)
    parser.add_argument('--eos_token_id', type=int, default=0)
    parser.add_argument('--feed_forward_proj', type=str, default='relu')
    parser.add_argument('--dataset_path', type=str, default='../data/Beauty/train.parquet')
    parser.add_argument('--code_path', type=str, default='../data/Beauty/Beauty_t5_rqvae.npy')
    parser.add_argument('--image_code_path', type=str, default='../data/Beauty/Beauty_t5_rqvae_image.npy')
    parser.add_argument('--save_path', type=str, default='./ckpt/pretrain.pth')
    parser.add_argument('--log_path', type=str, default='./logs/pretrain.log')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--max_len', type=int, default=20)
    
    args = parser.parse_args()
    config = vars(args)
    
    logging.basicConfig(
        filename=config['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    set_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    train_dataset = PretrainDataset(
        dataset_path=config['dataset_path'],
        code_path=config['code_path'],
        image_code_path=config['image_code_path'],
        mode='train',
        max_len=config['max_len'],
        mask_prob=0.2,
        base_offset_codebooks=4
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=pretrain_collate_fn
    )
    
    model = AVATARPretrain(config).to(device)
    logging.info(f"Model Parameters: {model.n_parameters}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    
    logging.info("Starting Pre-training...")
    for epoch in range(config['num_epochs']):
        loss = pretrain_epoch(model, train_loader, optimizer, device)
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {loss:.4f}")
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), config['save_path'])
            logging.info(f"Checkpoint saved to {config['save_path']}")

    logging.info("Pre-training Finished.")
