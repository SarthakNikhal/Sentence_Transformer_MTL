# Task 3: Training Considerations & Transfer Learning

This write-up explains key training strategies and how transfer learning was approached in the Multi-Task Learning (MTL) model.


### 🔒 Freezing Scenarios

1. Entire Network Frozen
- Purpose: Use the model as a static feature extractor.
- Benefit: Fast, avoids overfitting.
- Drawback: No domain or task adaptation.
- Use Case: Very limited training data.

---

2. Only Transformer Frozen
- Purpose: Keep pre-trained representations intact while training the heads.
- Benefit: Efficient training, stable features.
- Drawback: Tasks that differ from pretraining domain may underperform.
- Use Case: Medium-sized data or unrelated tasks.

---

3. Only One Head Frozen
- Purpose: Preserve performance of a previously trained task while adapting to a new one.
- Benefit: Useful in continual learning.
- Drawback: Encoder changes may harm frozen task.
- Use Case: When prioritizing one task over another.

---

### Transfer Learning Strategy

Pre-trained Model
- **Choice:** `distilbert-base-uncased` for general-purpose language understanding.

Freezing Approach
1. Initially froze the encoder.
2. Trained task-specific heads.
3. Gradually unfroze encoder for deeper fine-tuning.

### Rationale
- Lower layers capture syntax, best kept stable.
- Higher layers are more task-specific and benefit from fine-tuning.
- Gradual unfreezing prevents instability and catastrophic forgetting.

------------------------------------------------------------------------------

# Task 4: Training Loop Summary
This summary outlines the training loop for our Multi-Task Sentence Transformer.


### Training Design
- Used PyTorch to alternate between Task A (classification) and Task B (sentiment analysis).
- Each task used its own batch of sentence-label pairs.
- Model output routed through the correct task-specific head based on task ID.



### Loss Handling
- Used `CrossEntropyLoss` for both heads.
- Losses from both tasks were summed equally:
  
  ```python
  total_loss = loss_a + loss_b
