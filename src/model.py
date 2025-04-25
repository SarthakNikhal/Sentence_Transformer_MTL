from transformers import DistilBertModel
import torch
import torch.nn as nn

class SentenceTransformer(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super(SentenceTransformer, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        last_hidden_state = output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', 
                 num_classes_task_a=3, num_classes_task_b=3):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.encoder = SentenceTransformer(model_name)
        hidden_size = 768  # DistilBERT hidden size

        # Task-specific heads
        self.classification_head = nn.Linear(hidden_size, num_classes_task_a)
        self.sentiment_head = nn.Linear(hidden_size, num_classes_task_b)

    def forward(self, input_ids, attention_mask, task='A'):
        embeddings = self.encoder(input_ids, attention_mask)
        if task == 'A':
            return self.classification_head(embeddings)
        elif task == 'B':
            return self.sentiment_head(embeddings)
        else:
            raise ValueError("Task must be 'A' or 'B'")
