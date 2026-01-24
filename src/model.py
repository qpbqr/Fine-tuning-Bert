import torch
import torch.nn as nn
from transformers import BertModel
import config

class BERTClass(nn.Module):
    """
    BERT classification model for sentiment analysis task
    """
    def __init__(self, num_classes=2):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained(config.MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768, num_classes)  # BERT-base output dimension is 768
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: tokenized input ids
            attention_mask: attention mask
            
        Returns:
            logits: classification logits
        """
        output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # use [CLS] token output
        pooler_output = output.pooler_output
        output = self.dropout(pooler_output)
        output = self.linear(output)
        return output