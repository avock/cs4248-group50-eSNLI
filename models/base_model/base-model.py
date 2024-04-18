import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer

# Set up GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Custom Pytorch Dataset Class + Batch Dataloader for improved efficiency
"""
class NliDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

"""
Example of how to use the NliDataset class to create a DataLoader
"""

"Assuming you have X_train, X_test, y_train, y_test loaded as your training and test data"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings_1 = tokenizer(X_train['Sentence1'].astype(str).tolist(), truncation=True, padding=True)
train_encodings_2 = tokenizer(X_train['Sentence2'].astype(str).tolist(), truncation=True, padding=True)
train_encodings_3 = tokenizer(X_train['Explanation_1'].astype(str).tolist(), truncation=True, padding=True)
test_encodings_1 = tokenizer(X_test['Sentence1'].astype(str).tolist(), truncation=True, padding=True)
test_encodings_2 = tokenizer(X_test['Sentence2'].astype(str).tolist(), truncation=True, padding=True)
test_encodings_3 = tokenizer(X_test['Explanation_1'].astype(str).tolist(), truncation=True, padding=True)

# Create DataLoader
train_dataset_1 = NliDataset(train_encodings_1, y_train.tolist())
train_loader_1 = DataLoader(train_dataset_1, batch_size = 16)
train_dataset_2 = NliDataset(train_encodings_2, y_train.tolist())
train_loader_2 = DataLoader(train_dataset_2, batch_size = 16)
train_dataset_3 = NliDataset(train_encodings_3, y_train.tolist())
train_loader_3 = DataLoader(train_dataset_3, batch_size = 16)

test_dataset_1 = NliDataset(test_encodings_1, y_test.tolist())
test_dataset_2 = NliDataset(test_encodings_2, y_test.tolist())
test_dataset_3 = NliDataset(test_encodings_3, y_test.tolist())
test_loader_1 = DataLoader(test_dataset_1, batch_size = 16)
test_loader_2 = DataLoader(test_dataset_2, batch_size = 16)
test_loader_3 = DataLoader(test_dataset_3, batch_size = 16)

"""
InferSent Model Architecture with a BiLSTM Encoder and a MLPClassifier
"""
class BiLSTMEncoder(nn.Module):
    def __init__(self, hidden_dim, maxpool=False, batch_size=64):
        super(BiLSTMEncoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.maxpool = maxpool
        self.device = device
        self.emb_dim = 300

        self.embedding = nn.Embedding(30000, self.emb_dim)
        self.linear = nn.Linear(self.emb_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.projection = nn.Sequential(self.embedding, self.linear, self.relu)

        self.lstm = nn.LSTM(input_size=self.hidden_dim,
                            hidden_size=self.hidden_dim,
                            bidirectional=True,
                            batch_first=True)

    def forward(self, x):
        lengths = [len(sent) for sent in x]
        x = self.projection(x)

        h0 = torch.zeros(2, x.shape[0], self.hidden_dim).to(self.device)
        c0 = torch.zeros(2, x.shape[0], self.hidden_dim).to(self.device)

        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        padded_output, _ = self.lstm(x, (h0, c0))
        output, _ = nn.utils.rnn.pad_packed_sequence(padded_output,
                                                     batch_first=True)

        max_vecs = [torch.max(x, 0)[0] for x in output]
        embed = torch.stack(max_vecs, 0)
        return embed

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(Classifier, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.lin1, self.relu, self.lin2, self.relu, self.lin3)

    def forward(self, premise, hypothesis, explanation):
        combined = torch.cat((premise, hypothesis, torch.abs(premise - hypothesis), torch.abs(premise - explanation), torch.abs(hypothesis - explanation),
                              premise * explanation, hypothesis * explanation,
                              premise * hypothesis), 1)
        out = self.net(combined)
        return out

class InferSent(nn.Module):
    def __init__(self, enc_hidden_dim, cls_hidden_dim, train_loader_1, train_loader_2, train_loader_3, test_loader_1, test_loader_2, test_loader_3, num_epochs=2):
        super(InferSent, self).__init__()
        self.encoder = BiLSTMEncoder(enc_hidden_dim, maxpool=True)
        self.cls_input_dim = enc_hidden_dim * 2 * 4
        self.classifier = Classifier(self.cls_input_dim, cls_hidden_dim, out_dim=3)
        self.train_loader_1 = train_loader_1
        self.train_loader_2 = train_loader_2
        self.train_loader_3 = train_loader_3
        self.test_loader_1 = test_loader_1
        self.test_loader_2 = test_loader_2
        self.test_loader_3 = test_loader_3
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, batch):
        (premise, hypothesis, explanation) = batch
        premise_encoded = self.encoder(premise)
        hypothesis_encoded = self.encoder(hypothesis)
        explanation_encoded = self.encoder(explanation)
        out = self.classifier(premise_encoded, hypothesis_encoded, explanation_encoded)
        return out

    def train_model(self, optimizer, scheduler):
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.num_epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, self.num_epochs))
            print('Training...')
            self.train()
            running_loss = 0.0
            
            for step, (batch1, batch2, batch3) in enumerate(zip(self.train_loader_1, self.train_loader_2, self.train_loader_3)):
                b_input_ids = batch1['input_ids'].to(self.device)
                b_input_mask = batch1['attention_mask'].to(self.device)
                b_labels = batch1['labels'].to(self.device)
                b_input_ids2 = batch2['input_ids'].to(self.device)
                b_input_mask2 = batch2['attention_mask'].to(self.device)
                b_labels2 = batch2['labels'].to(self.device)
                b_input_ids3 = batch3['input_ids'].to(self.device)
                b_input_mask3 = batch3['attention_mask'].to(self.device)
                b_labels3 = batch3['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self((b_input_ids, b_input_ids2, b_input_ids3))
                loss = criterion(outputs, b_labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
                
            avg_loss = running_loss / len(self.train_loader_1)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss}')
            
    def evaluate_model(self):
        self.eval()
        predictions, true_labels = [], []
        for batch1, batch2, batch3 in zip(self.test_loader_1, self.test_loader_2, self.test_loader_3):
            b1 = batch1['input_ids'].to(self.device)
            b2 = batch2['input_ids'].to(self.device)
            b3 = batch3['input_ids'].to(self.device)
            with torch.no_grad():
                outputs = self((b1, b2, b3))
            logits = outputs.detach().cpu().numpy()
            label_ids = batch1['labels'].to('cpu').numpy()
            predictions.append(logits)
            true_labels.append(label_ids)

        predictions = np.argmax(np.concatenate(predictions, axis=0), axis=1)
        true_labels = np.concatenate(true_labels, axis=0)

        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

        print("Evaluation Results:")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

model = InferSent(1028, 512, train_loader_1, train_loader_2, train_loader_3, test_loader_1, test_loader_2, test_loader_3).to(device)

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
total_steps = len(train_loader_1) * 5
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)
model.train_model(optimizer, scheduler)
model.evaluate_model()
