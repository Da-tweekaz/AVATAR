import torch
from torch.utils.data import DataLoader

class GenRecDataLoader(DataLoader):
    """
    GenRecDataLoader for Generative Recommendation tasks.
    
    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): Number of subprocesses to use for data loading.
        collate_fn (callable, optional): Function to merge a list of samples to form a mini-batch.
        worker_init_fn (callable, optional): Function to initialize workers.
    """
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=None, worker_init_fn=None):
        collate_fn = self.collate_fn
        super(GenRecDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers, collate_fn=collate_fn, worker_init_fn=worker_init_fn)
    
            
    def collate_fn(self, batch, pad_token=0):
        """
        Create attention mask for input sequence.
        
        Args:
            batch (list): List of samples from the dataset.
        
        Returns:
            dict: Batched data with padded sequences.
        """
        history_texts = [item['history_text'] for item in batch]
        history_images = [item['history_image'] for item in batch]
        targets = [item['target'] for item in batch]

        flattened_history_texts = torch.stack(
            [torch.tensor([elem for sublist in history for elem in sublist], dtype=torch.int64) for history in history_texts]
        )
        
        flattened_history_images = torch.stack(
            [torch.tensor([elem for sublist in history for elem in sublist], dtype=torch.int64) for history in history_images]
        )

        flattened_targets = torch.stack(
            [torch.tensor(target, dtype=torch.int64) for target in targets]
        )

        attention_masks = torch.stack(
            [torch.tensor([1 if elem != pad_token else 0 for elem in h], dtype=torch.int64) for h in flattened_history_texts]
        )

        image_attention_masks = torch.stack(
            [torch.tensor([1 if elem != pad_token else 0 for elem in h], dtype=torch.int64) for h in flattened_history_images]
        )

        return {
            'history_text': flattened_history_texts, 
            'history_image': flattened_history_images,
            'target': flattened_targets, 
            'attention_mask': attention_masks,
            'image_attention_mask': image_attention_masks
        }
