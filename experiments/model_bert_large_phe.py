import pandas as pd
import time, datetime, numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

"""## Mounting Google Drive to Collab"""

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/CS4248/esnli_train.csv')
test = pd.read_csv('/content/drive/MyDrive/CS4248/esnli_test.csv')

df.shape

df.head()

"""## Utility Functions"""

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def select_cols(df, col_list):
    '''
    Select columns from a dataframe
    '''
    return df[col_list]

"""## Data Pre-processing Utility Functions"""

def combine_sentences(df, col_list):

    # start_token = '[CLS]'
    # seperator = '[SEP]'

    # combined_text_list = [f'{start_token}{s1}{seperator}{s2}' for s1, s2 in zip(sentence1, sentence2)]
    # return combined_text_list

    results_df = df.copy()

    results_df['combined_text'] = '[CLS]' + results_df[col_list].astype(str).agg('[SEP]'.join, axis=1)

    return results_df

"""Train/Test Input Data Handling"""

# target_cols = ['Sentence1', 'Sentence2', 'gold_label'] # Premise, Hypothesis
target_cols = ['Sentence1', 'Sentence2', 'Explanation_1', 'gold_label'] # Premise, Hypothesis, Explanation
# target_cols = ['Sentence2', 'Sentence1', 'gold_label'] # Hypothesis, Premise
# target_cols = ['Sentence1', 'Explanation_1', 'Sentence2', 'gold_label'] # Premise, Explanation, Hypothesis

df = select_cols(df, target_cols)
test_df = select_cols(test, target_cols)

df.head()

df = combine_sentences(df, target_cols[:-1])
test_df = combine_sentences(test_df, target_cols[:-1])

df.head()

lables = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2
}

df['labels'] = df['gold_label'].map(lables)
test_df['labels'] = test_df['gold_label'].map(lables)

df.head()

df['combined_text'][0]

X = df['combined_text']
y = df['labels']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

X_train.shape, X_val.shape, y_train.shape, y_val.shape

X_test = test_df['combined_text']
y_test = test_df['labels']

X_test.shape, y_test.shape

class NliDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# tokenize train/validation data
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(X_val.tolist(), truncation=True, padding=True)

# tokenize test data
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)

# creating Dataset objects
train_dataset = NliDataset(train_encodings, y_train.tolist())
val_dataset = NliDataset(val_encodings, y_val.tolist())
test_dataset = NliDataset(test_encodings, y_test.tolist())

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = BertForSequenceClassification.from_pretrained(
    "bert-large-uncased",
    num_labels = 3,
    output_attentions = False,
    output_hidden_states = False,
)

#unfreezing layer 11 and the classifier. note: the pooler is still frozen
for name, param in model.named_parameters():
    if 'classifier' not in name and '11' not in name:
        param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if torch.cuda.is_available():
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")

print("Current device model is on:", model.device)

optimizer = AdamW(model.parameters(),
                  lr = 5e-5,
                  eps = 1e-8
                 )

epochs = 2
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

# Begin Training Loop
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

print("")
print("Training complete!")

model.eval()

predictions, true_labels = [], []

# load test cases into GPU in batches
for batch in test_loader:
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    label_ids = batch['labels'].to('cpu').numpy()

    predictions.append(logits)
    true_labels.append(label_ids)

predictions = np.argmax(np.concatenate(predictions, axis=0), axis=1)
true_labels = np.concatenate(true_labels, axis=0)

accuracy = accuracy_score(true_labels, predictions)
print("Accuracy:", accuracy)

precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}")

model_save_path = "/content/drive/MyDrive/CS4248/model_large_PHE.pth"
optimizer_save_path = "/content/drive/MyDrive/CS4248/model_large_PHE.pth"

# Save the model, optimizer and encodings state
torch.save(model.state_dict(), model_save_path)
torch.save(optimizer.state_dict(), optimizer_save_path)