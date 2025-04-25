import torch
import torch.nn as nn
import torch.optim as optim

from model import MultiTaskSentenceTransformer
from transformers import AutoTokenizer


# Fake sentences
sentences = [
    "Artificial Intelligence is evolving rapidly.",  # Tech (Task A), Positive (Task B)
    "The government passed a new healthcare law.",    # Politics (Task A), Neutral (Task B)
]
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')


# Tokenize
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Initialize model
mtl_model = MultiTaskSentenceTransformer()

# forward pass classification task b
logits_task_a = mtl_model(inputs['input_ids'], inputs['attention_mask'], task='A')
print("Task A (Classification) Logits:\n", logits_task_a)

# Forward pass Sentiment analysis task b
logits_task_b = mtl_model(inputs['input_ids'], inputs['attention_mask'], task='B')
print("Task B (Sentiment) Logits:\n", logits_task_b)


# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(mtl_model.parameters(), lr=2e-5)

# Fake data
task_a_data = [("AI is the future.", 0), ("Politics is complex.", 2)]
task_b_data = [("This is great!", 0), ("That was awful.", 1)]

# Dummy label mapping
def preprocess(data):
    sentences, labels = zip(*data)
    inputs = tokenizer(list(sentences), padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(labels)
    return inputs['input_ids'], inputs['attention_mask'], labels

# Training
for epoch in range(10):  # Can be more
    mtl_model.train()
    
    #task A
    input_ids, attention_mask, labels_a = preprocess(task_a_data)
    logits_a = mtl_model(input_ids, attention_mask, task='A')
    loss_a = loss_fn(logits_a, labels_a)
    
    #tASK 
    input_ids, attention_mask, labels_b = preprocess(task_b_data)
    logits_b = mtl_model(input_ids, attention_mask, task='B')
    loss_b = loss_fn(logits_b, labels_b)
    
    total_loss = loss_a + loss_b  # Simple sum; can be weighted
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch} - Loss A: {loss_a.item():.4f}, Loss B: {loss_b.item():.4f}")
