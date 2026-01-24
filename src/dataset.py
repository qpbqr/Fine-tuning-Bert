import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import config
import pandas as pd

class FinancialNewsDataset(Dataset):
    """
    Financial news sentiment analysis dataset
    """
    def __init__(self, dataframe, tokenizer, max_len):
        """
        Args:
            dataframe: DataFrame containing headline and label columns
            tokenizer: BERT tokenizer
            max_len: maximum sequence length
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.headline
        self.targets = dataframe.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'mask': inputs['attention_mask'].flatten(),
            'targets': torch.tensor(self.targets.iloc[index], dtype=torch.long)
        }