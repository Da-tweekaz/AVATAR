import torch
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import random
from tqdm import tqdm
import logging
from dataset import GenRecDataset
from dataloader import GenRecDataLoader

class AVATAR_Finetune(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(AVATAR_Finetune, self).__init__()
        
        t5config = T5Config(
            num_layers=config['num_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            vocab_size=self.vocab_size,
            pad_token_id=config['pad_token_id'],
            eos_token_id=config['eos_token_id'],
            decoder_start_token_id=config['pad_token_id'],
            feed_forward_proj=config['feed_forward_proj'],
        )

        self.vocab_size = config['vocab_size']
        self.image_vocab_size = config['image_vocab_size']
        self.d_model = config['d_model']
        self.prefix_length = config['prefix_length']
        self.prefix_dropout = config['prefix_dropout']
        self.model = T5ForConditionalGeneration(t5config)
        self.image_embeddings = nn.Embedding(self.image_vocab_size, self.d_model)
        self.shared_encoder = self.model.encoder

        projection_dim = config.get('projection_dim', self.d_model)

        self.text_proj = nn.Linear(self.d_model, projection_dim)
        self.image_proj = nn.Linear(self.d_model, projection_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.prefix_map = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model * self.prefix_length)
        )
        self.prefix_norm = nn.LayerNorm(self.d_model)
        self.prefix_drop = nn.Dropout(self.prefix_dropout)

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

    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _build_prefix(self, image_input_ids: torch.Tensor, image_attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs_embeds_image = self.image_embeddings(image_input_ids)
        encoder_outputs_image = self.shared_encoder(
            inputs_embeds=inputs_embeds_image,
            attention_mask=image_attention_mask
        )
        last_hidden_state_image = encoder_outputs_image.last_hidden_state
        image_feat = self._mean_pooling(last_hidden_state_image, image_attention_mask)
        prefix_flat = self.prefix_map(image_feat)
        prefix_tokens = prefix_flat.view(image_feat.size(0), self.prefix_length, self.d_model)
        prefix_tokens = self.prefix_norm(prefix_tokens)
        prefix_tokens = self.prefix_drop(prefix_tokens)
        prefix_mask = torch.ones((image_feat.size(0), self.prefix_length), device=image_feat.device, dtype=image_attention_mask.dtype)
        return prefix_tokens, prefix_mask

    def forward(self, input_ids: torch.Tensor, image_input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, 
                image_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        """
        Forward pass handling both Generation and Contrastive Learning.
        
        Args:
            input_ids: Text ID sequences (Batch, Seq)
            image_input_ids: Image ID sequences (Batch, Seq)
            attention_mask: Mask for Text
            image_attention_mask: Mask for Image
            labels: Target Text IDs for generation
        """
        
        prefix_tokens, prefix_mask = self._build_prefix(image_input_ids, image_attention_mask)
        
        inputs_embeds_text = self.model.shared(input_ids)
        
        inputs_embeds = torch.cat([prefix_tokens, inputs_embeds_text], dim=1)
        attn_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels
        )
        loss_gen = outputs.loss
        logits_gen = outputs.logits
        
        inputs_embeds_text = self.model.shared(input_ids)
        
        encoder_outputs_text = self.shared_encoder(
            inputs_embeds=inputs_embeds_text,
            attention_mask=attention_mask
        )
        last_hidden_state_text = encoder_outputs_text.last_hidden_state
        
        inputs_embeds_image = self.image_embeddings(image_input_ids)
        
        encoder_outputs_image = self.shared_encoder(
            inputs_embeds=inputs_embeds_image,
            attention_mask=image_attention_mask
        )
        last_hidden_state_image = encoder_outputs_image.last_hidden_state
        
        text_feat = self._mean_pooling(last_hidden_state_text, attention_mask)
        image_feat = self._mean_pooling(last_hidden_state_image, image_attention_mask)
        
        text_feat = self.text_proj(text_feat)
        image_feat = self.image_proj(image_feat)
        
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
        image_feat = image_feat / image_feat.norm(dim=1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits_per_text = logit_scale * text_feat @ image_feat.t()
        logits_per_image = logits_per_text.t()
        
        batch_size = input_ids.shape[0]
        labels_cl = torch.arange(batch_size, device=input_ids.device)
        
        loss_cl_text = nn.CrossEntropyLoss()(logits_per_text, labels_cl)
        loss_cl_image = nn.CrossEntropyLoss()(logits_per_image, labels_cl)
        loss_cl = (loss_cl_text + loss_cl_image) / 2
        
        return loss_gen, loss_cl, logits_gen
    
    def generate(self, input_ids: torch.Tensor, 
                 image_input_ids: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None, 
                 image_attention_mask: Optional[torch.Tensor] = None,
                 num_beams: int = 20, 
                 **kwargs):
        """Generate recommendations using the model."""
        
        prefix_tokens, prefix_mask = self._build_prefix(image_input_ids, image_attention_mask)
        inputs_embeds_text = self.model.shared(input_ids)
        inputs_embeds = torch.cat([prefix_tokens, inputs_embeds_text], dim=1)
        attn_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_length=5,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            **kwargs
        )

def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert (
        preds.shape[1] == maxk
    ), f'preds.shape[1] = {preds.shape[1]} != {maxk}'

    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
      cur_label = labels[i].tolist()
      for j in range(maxk):
        cur_pred = preds[i, j].tolist()
        if cur_pred == cur_label:
          pos_index[i, j] = True
          break
    return pos_index

def recall_at_k(pos_index, k):
  return pos_index[:, :k].sum(dim=1).cpu().float()

def ndcg_at_k(pos_index, k):
  ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
  dcg = 1.0 / torch.log2(ranks + 1)
  dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
  return dcg[:, :k].sum(dim=1).cpu().float()

def train(model, train_loader, optimizer, device, lambda_cl=0.1):
    model.train()
    total_loss = 0.0
    total_gen_loss = 0.0
    total_cl_loss = 0.0
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['history_text'].to(device)
        image_input_ids = batch['history_image'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image_attention_mask = batch['image_attention_mask'].to(device)
        labels = batch['target'].to(device)

        optimizer.zero_grad()
        
        loss_gen, loss_cl, _ = model(
            input_ids=input_ids, 
            image_input_ids=image_input_ids,
            attention_mask=attention_mask, 
            image_attention_mask=image_attention_mask,
            labels=labels
        )
        
        loss = loss_gen + lambda_cl * loss_cl
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_gen_loss += loss_gen.item()
        total_cl_loss += loss_cl.item()
        
    return total_loss / len(train_loader)

def evaluate(model, eval_loader, topk_list, beam_size, device):
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['history_text'].to(device)
            image_input_ids = batch['history_image'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_attention_mask = batch['image_attention_mask'].to(device)
            labels = batch['target'].to(device)

            preds = model.generate(
                input_ids=input_ids, 
                image_input_ids=image_input_ids,
                attention_mask=attention_mask, 
                image_attention_mask=image_attention_mask,
                num_beams=beam_size
            )
            preds = preds[:, 1:]
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)
            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
            
            for k in topk_list:
                recall = recall_at_k(pos_index, k).mean().item()
                ndcg = ndcg_at_k(pos_index, k).mean().item()
                recalls['Recall@' + str(k)].append(recall)
                ndcgs['NDCG@' + str(k)].append(ndcg)
                
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    """
    Initialize worker seed for DataLoader.
    """

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVATAR Fine-tuning configuration")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--infer_size', type=int, default=96, help='Inference size for generating recommendations')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (e.g., "cuda" or "cpu")')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the model')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers in the model')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model')
    parser.add_argument('--d_ff', type=int, default=1024, help='Dimension of the feed-forward layer')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--d_kv', type=int, default=64, help='Dimension of key and value vectors')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--vocab_size', type=int, default=1025, help='Vocabulary size')
    parser.add_argument('--image_vocab_size', type=int, default=1025, help='Vocabulary size for image')
    parser.add_argument('--pad_token_id', type=int, default=0, help='Padding token ID')
    parser.add_argument('--eos_token_id', type=int, default=0, help='End of sequence token ID')
    parser.add_argument('--feed_forward_proj', type=str, default='relu', help='Feed forward projection type')
    parser.add_argument('--max_len', type=int, default=20, help='Maximum length for padding or truncation')
    parser.add_argument('--dataset_path', type=str, default='../data/Beauty', help='Path to the dataset')
    parser.add_argument('--code_path', type=str, default='../data/Beauty/Beauty_t5_rqvae.npy', help='Path to the item-to-code mapping file')
    parser.add_argument('--image_code_path', type=str, default='../data/Beauty/Beauty_t5_rqvae_image.npy', help='Path to the item-to-image-code mapping file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluation'], help='Mode of operation')
    parser.add_argument('--log_path', type=str, default='./logs/finetune.log', help='Path to the log file')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
    parser.add_argument('--save_path', type=str, default='./ckpt/finetune.pth', help='Path to save the trained model')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--topk_list', type=list, default=[5,10,20], help='List of top-k values for evaluation metrics')
    parser.add_argument('--beam_size', type=int, default=30, help='Beam size for generation')
    parser.add_argument('--lambda_cl', type=float, default=0.1, help='Weight for contrastive loss')
    parser.add_argument('--projection_dim', type=int, default=128, help='Dimension of projection head for contrastive learning')
    parser.add_argument('--pretrained_path', type=str, default='./ckpt/pretrain.pth', help='Path to the pre-trained model')
    parser.add_argument('--prefix_length', type=int, default=8, help='Length of the image prefix')
    parser.add_argument('--prefix_dropout', type=float, default=0.1, help='Dropout rate for the prefix')

    config = vars(parser.parse_args())

    logging.basicConfig(
        filename=config['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Configuration: {config}")
    
    set_seed(config['seed'])

    model = AVATAR_Finetune(config)
    print(model.n_parameters)
    logging.info(model.n_parameters)

    if os.path.exists(config['pretrained_path']):
        try:
            pretrained_state = torch.load(config['pretrained_path'], weights_only=True)
            model_state = model.state_dict()
            filtered_state = {k: v for k, v in pretrained_state.items() 
                            if k in model_state and v.shape == model_state[k].shape}
            model.load_state_dict(filtered_state, strict=False)
            
            if 'model.shared.weight' in pretrained_state:
                shared_weight = pretrained_state['model.shared.weight']
                with torch.no_grad():
                    model.model.shared.weight.copy_(shared_weight[:model.vocab_size])
                    model.image_embeddings.weight[0].copy_(shared_weight[0])
                    model.image_embeddings.weight[1 : 1025].copy_(shared_weight[1025 : 1025 + 1024])
                
        except Exception as e:
            logging.error(f"Error loading pre-trained weights: {e}")
    else:
        logging.warning(f"Pre-trained model path {config['pretrained_path']} not found.")

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    train_dataset = GenRecDataset(
        dataset_path=config['dataset_path']+ '/train.parquet',
        code_path=config['code_path'],
        image_code_path=config['image_code_path'],
        mode='train',
        max_len=config['max_len'],
        base_offset_codebooks=0
    )
    validation_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/valid.parquet',
        code_path=config['code_path'],
        image_code_path=config['image_code_path'],
        mode='evaluation',
        max_len=config['max_len'],
        base_offset_codebooks=0
    )
    test_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/test.parquet',
        code_path=config['code_path'],
        image_code_path=config['image_code_path'],
        mode='evaluation',
        max_len=config['max_len'],
        base_offset_codebooks=0
    )

    train_dataloader = GenRecDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, worker_init_fn=worker_init_fn)
    validation_dataloader = GenRecDataLoader(validation_dataset, batch_size=config['infer_size'], shuffle=False, worker_init_fn=worker_init_fn)
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False, worker_init_fn=worker_init_fn)
    
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    best_ndcg = 0.0
    early_stop_counter = 0
    
    for epoch in range(config['num_epochs']):
        logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        train_loss = train(model, train_dataloader, optimizer, device, lambda_cl=config['lambda_cl'])
        logging.info(f"Training loss: {train_loss}")

        avg_recalls, avg_ndcgs = evaluate(model, validation_dataloader, config['topk_list'], config['beam_size'], device)
        logging.info(f"Validation Dataset: {avg_recalls}")
        logging.info(f"Validation Dataset: {avg_ndcgs}")
        if avg_ndcgs['NDCG@20'] > best_ndcg:
            best_ndcg = avg_ndcgs['NDCG@20']
            early_stop_counter = 0
            test_avg_recalls, test_avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)
            logging.info(f"Best NDCG@20: {best_ndcg}")
            logging.info(f"Test Dataset: {test_avg_recalls}")
            logging.info(f"Test Dataset: {test_avg_ndcgs}")

            torch.save(model.state_dict(), config['save_path'])
            logging.info(f"Best model saved to {config['save_path']}")
        else:
            early_stop_counter += 1
            logging.info(f"No improvement in NDCG@20. Early stop counter: {early_stop_counter}")
            if early_stop_counter >= config['early_stop']:
                logging.info("Early stopping triggered.")
                break
