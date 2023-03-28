from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, SubsetRandomSampler, Dataset, DataLoader
from transformers import AdamW, BertForSequenceClassification
from sklearn.model_selection import train_test_split

# Define the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# Set random seeds for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

df = pd.read_csv('IMDB Dataset.csv')

class CustomDataset(Dataset):
    def __init__(self, input_ids, labels, indices):
        self.input_ids = input_ids
        self.labels = labels
        self.indices = indices

    def __getitem__(self, index):
        if index not in self.indices:
            # Skip this sample
            return None
        input_id = self.input_ids[index]
        label = self.labels[index]
        return input_id, label

    def __len__(self):
        return len(self.labels)
# Define the data fields
TEXT = df['review']
LABEL = df['sentiment'].apply(lambda x: 1 if x=='positive' else 0)

# Tokenize the input texts
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(TEXT.tolist(), padding=True, truncation=True, return_tensors="pt")

# Apply normalization
inputs['input_ids'] = inputs['input_ids'] / tokenizer.vocab_size
print(inputs['input_ids'])
# Split the data into train and test sets
train_indices, test_indices = train_test_split(np.arange(len(LABEL)), test_size=0.2, random_state=42)
train_dataset = CustomDataset(inputs['input_ids'], LABEL, train_indices)
test_dataset = CustomDataset(inputs['input_ids'], LABEL, test_indices)

# Define the data samplers
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
test_loader = DataLoader(test_dataset, batch_size=16, sampler=test_sampler)

# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()
# Train the model
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        if torch.isnan(inputs).any():
            # Skip this batch
            continue
        optimizer.zero_grad()
        inputs = inputs.long()  # convert to Long tensor
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Training loss: {running_loss/len(train_loader)}")


# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Test accuracy: {correct/total}")
