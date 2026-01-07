import pandas as pd
import numpy as np
from torch.utils.data import Dataset

def process_data(file_path, mode, max_len, PAD_TOKEN=0):
    """
    Process parquet data based on mode ('train' or 'evaluation').

    Args:
        file_path (str): Path to the parquet file.
        mode (str): Mode of operation ('train' or 'evaluation').
        max_len (int): Maximum length for padding or truncation.

    Returns:
        list: Processed data.
    """

    data = pd.read_parquet(file_path)

    data['sequence'] = data['history'].apply(lambda x: list(x)) + data['target'].apply(lambda x: [x])

    if mode == 'train':
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            for i in range(1, len(sequence)):
                processed_data.append({
                    'history': sequence[:i],
                    'target': sequence[i]
                })
    elif mode == 'evaluation':
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            processed_data.append({
                'history': sequence[:-1],
                'target': sequence[-1]
            })
    else:
        raise ValueError("Mode must be 'train' or 'evaluation'.")

    for item in processed_data:
        item['history'] = pad_or_truncate(item['history'], max_len)

    return processed_data

def pad_or_truncate(sequence, max_len, PAD_TOKEN=0):
    """
    Pad or truncate a sequence to a specified maximum length.

    Args:
        sequence (list): Input sequence.
        max_len (int): Maximum length for the sequence.

    Returns:
        list: Padded or truncated sequence.
    """
    if len(sequence) > max_len:
        return sequence[-max_len:]
    else:
        return [PAD_TOKEN] * (max_len - len(sequence)) + sequence
    
def item2code(code_path, codebook_size=256, base_offset_codebooks=0):
    """
    Convert itemID to code
    :param code_path: npy file path to store rqvae codes
    :param codebook_size: size of the codebook
    :param base_offset_codebooks: number of codebooks to offset by
    :return: dict item_to_code, code_to_item
    """
    data = np.load(code_path, allow_pickle=True)
    item_to_code = {}
    code_to_item = {}
    
    for index, code in enumerate(data):
        offsets = [c + (i + base_offset_codebooks) * codebook_size + 1 for i,c in enumerate(code)]
        item_to_code[index + 1] = offsets
        code_to_item[tuple(offsets)] = index + 1

    return item_to_code, code_to_item

class GenRecDataset(Dataset):
    def __init__(self, dataset_path, code_path, image_code_path, mode, max_len, PAD_TOKEN=0, base_offset_codebooks=0):
        """
        Initialize the GenRecDataset.
        Args:
            dataset_path (str): Path to the dataset file.
            code_path (str): Path to the item-to-code mapping file.
            image_code_path (str): Path to the image code file.
            mode (str): Mode of operation ('train' or 'evaluation').
            max_len (int): Maximum length for padding or truncation.
            PAD_TOKEN (int, optional): Token used for padding. Defaults to 0.
            base_offset_codebooks (int, optional): Offset for image codebooks. Defaults to 0.
        """
        self.dataset_path = dataset_path
        self.code_path = code_path
        self.image_code_path = image_code_path
        self.mode = mode
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        self.base_offset_codebooks = base_offset_codebooks

        self.item_to_code, self.code_to_item = item2code(code_path, 256, base_offset_codebooks=0)
        self.item_to_image_code, self.image_code_to_item = item2code(image_code_path, 256, base_offset_codebooks=self.base_offset_codebooks)
            
        self.data = self._prepare_data()

    def _prepare_data(self):
        """
        Process the dataset and convert items to codes.
        Returns:
            list: Processed data with items converted to codes.
        """

        processed_data = process_data(
            self.dataset_path, self.mode, self.max_len, self.PAD_TOKEN
        )

        for item in processed_data:

            history_item_ids = item['history']
        
            text_codes = [self.item_to_code.get(x, [self.PAD_TOKEN]*4) for x in history_item_ids]
            image_codes = [self.item_to_image_code.get(x, [self.PAD_TOKEN]*4) for x in history_item_ids]
            
            item['history_text'] = text_codes
            item['history_image'] = image_codes
            
            item['target'] = self.item_to_code.get(item['target'], [self.PAD_TOKEN]*4)
            
            del item['history']
            
        return processed_data
    
    def __getitem__(self, index):
        """
        Get a single data item by index.
        Args:
            index (int): Index of the data item.
        Returns:
            dict: A dictionary containing 'history_text', 'history_image' and 'target'.
        """
        return self.data[index]
    
    def __len__(self):
        """
        Get the total number of data.
        Returns:
            int: Total number of data.
        """
        return len(self.data)
