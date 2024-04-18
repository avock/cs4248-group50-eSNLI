import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader

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
# tokenizing data
tokenizer = BertTokenizer.from_pretrained('roberta-base')
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(X_val.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)

# creating Dataset objects
train_dataset = NliDataset(train_encodings, y_train.tolist())
val_dataset = NliDataset(val_encodings, y_val.tolist())
test_dataset = NliDataset(test_encodings, y_test.tolist())

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

"""
Model Initialization
"""
model = BertForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels = 3,
    output_attentions = False,
    output_hidden_states = False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# # uncomment when loading saved models

# model_type = 'premise_hypothesis_explanation'
# model_save_path = f"/content/drive/MyDrive/[CS4248] Project Folder/models/{model_type}.pth"
# optimizer_save_path = f"/content/drive/MyDrive/[CS4248] Project Folder/optimizer/{model_type}.pth"

# model.load_state_dict(torch.load(model_save_path))
# # optimizer.load_state_dict(torch.load(optimizer_save_path))

optimizer = AdamW(model.parameters(),
                  lr = 5e-5,
                  eps = 1e-8
                 )
epochs = 2
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)


"""
Model Training
"""
loss_values = []
for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # time taken for each epoch
    t0 = time.time()

    total_loss = 0

    model.train()

    for step, batch in enumerate(train_loader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        # clear previously calculated gradient before backward pass
        model.zero_grad()

        # forward pass
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        loss = outputs.loss

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss.item()

        # backwar pass
        loss.backward()

        # Clip the norm of the gradients to 1.0, helps prevents "exploding gradient"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

print("Training complete!")


"""
Model Evaluation
"""
model.eval()

predictions, true_labels, attention_maps, tokens_list = [], [], [], []

# load test cases into GPU in batches
for batch in test_loader:
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits.detach().cpu().numpy()
    label_ids = batch['labels'].to('cpu').numpy()

    predictions.append(logits)
    true_labels.append(label_ids)

predictions = np.argmax(np.concatenate(predictions, axis=0), axis=1)
true_labels = np.concatenate(true_labels, axis=0)

accuracy = accuracy_score(true_labels, predictions)
print("Accuracy:", accuracy)

precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}")