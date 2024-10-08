# %%
import preprocess as pre

# %%
print(pre.input_train[0:10])

# %% [markdown]
# ## pretrained BERT Model 
# 

# %%
from transformers import BertModel, BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)


# %% [markdown]
# ## define extraction function with cls

# %%
def extract_features_cls(sentences):
    # Tokenize the input sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Get the outputs from the BERT model
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)

    # The last hidden state is the first element of outputs
    last_hidden_state = outputs.last_hidden_state

    # Extract the [CLS] token for each sentence (index 0)
    cls_embeddings = last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]

    return cls_embeddings


# %%
features_cls = extract_features_cls(pre.input_train[0:10])

# %%
print(features_cls.shape)

# %% [markdown]
# # extractions fearture with 4 final layers 

# %%
def extract_features_4_layers(sentences):
    # Tokenize the input sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # Get the outputs from the BERT model
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)

    # The 4 last hidden state is the first element of outputs
    last_hidden_state = outputs.last_hidden_state # Shape: [batch_size
    h_s= outputs.hidden_states
    h_s_12 = h_s[12]
    h_s_11 = h_s[11]
    h_s_10 = h_s[10]
    h_s_9 = h_s[9]
    

    # Concatenate embeddings for each token
    # You'll want to concatenate outputs from each of the last four layers for each token
    # Each hidden layer output shape: (batch_size, sequence_length, hidden_size)
    concatenated_embeddings = torch.cat((h_s_12, h_s_11, h_s_10, h_s_9), dim=1)   # shape = (batch_size, sequence_length, hidden_size * 4)
    
    # Average the concatenated embeddings across the tokens for each sample
    # We can ignore the embedding for [CLS] and [SEP] tokens or average all if required
    # Here, let's average across all tokens
    average_embedding = torch.mean(concatenated_embeddings, dim=1)  # Resulting shape will be [10, 768]

    return average_embedding


# %%
features_extraction_4_layers = extract_features_4_layers(pre.input_train[0:10])

# %% [markdown]
# ## Create a Simple Neural Network

# %%
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# %% [markdown]
# ## Training the Model with cls feature extraction

# %%

# Initialize the model, loss function, and optimizer
model_cls = SimpleNN(input_size=768, num_classes=2)
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with cls
model_cls.train()
for epoch in range(10):  # number of epochs
    optimizer_cls.zero_grad()
    outputs_cls = model_cls(features_cls)
    loss_cls = criterion_cls(outputs_cls, pre.labels_tensor[0:10])
    loss_cls.backward()
    optimizer_cls.step()
    print(f'Epoch [{epoch + 1}/10], Loss: {loss_cls.item():.4f}')

# %% [markdown]
# ## Training the Model with 4 last hidden layers

# %%
# Initialize the model, loss function, and optimizer
model_4_layers = SimpleNN(input_size=768, num_classes=2)
criterion_4_layers = nn.CrossEntropyLoss()
optimizer_4_layers = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with cls
model_4_layers.train()
for epoch in range(10):  # number of epochs
    optimizer_4_layers.zero_grad()
    outputs_4_layers = model_4_layers(features_extraction_4_layers)
    loss_4_layers = criterion_4_layers(outputs_4_layers, pre.labels_tensor[0:10])
    loss_4_layers.backward()
    optimizer_4_layers.step()
    print(f'Epoch [{epoch + 1}/10], Loss: {loss_4_layers.item():.4f}')

# %% [markdown]
# ## Evaluating the Model

# %%



