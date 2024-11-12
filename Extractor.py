
from preproces import DataProcessor
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer





#Handles BERT model initialization and feature extraction methods.
class BERTFeatureExtractor:
    def __init__(self, model_name='bert-base-multilingual-cased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        
        
    # responsible for extracting features from the input sentences using BERT with the final layer.
    def extract_features_cls(self, sentences):
        # Tokenize the input sentences
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # Get the outputs from the BERT model
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = self.model(**inputs)
        # The last hidden state is the first element of outputs
        last_hidden_state = outputs.last_hidden_state
        # Extract the [CLS] token for each sentence (index 0)
        cls_embeddings = last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]
        return cls_embeddings

    
    # responsible for extracting features from the input sentences using BERT with the 4 last layers.
    def extract_features_4_layers(self, sentences):
        # Tokenize the input sentences
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # Get the outputs from the BERT model
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = self.model(**inputs)
        # Extract the last hidden states
        h_s = outputs.hidden_states
        h_s_12 = h_s[12]
        h_s_11 = h_s[11]
        h_s_10 = h_s[10]
        h_s_9 = h_s[9]
        
        # Concatenate embeddings for each token
        concatenated_embeddings = torch.cat((h_s_12, h_s_11, h_s_10, h_s_9), dim=1)  # shape = (batch_size, sequence_length, hidden_size  4)
        # Average the concatenated embeddings across the tokens for each sample
        average_embedding = torch.mean(concatenated_embeddings, dim=1)  # Resulting shape will be [batch_size, hidden_size  4]
        return average_embedding
    
    
    
#Defines a simple feedforward neural network.
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
    
    
    
# Manages the training process for the models.    
# class encapsulates the training loop, allowing for easy reuse and modification.
class ModelTrainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, features, labels, epochs=10):
        self.model.train()
        for epoch in range(epochs):  # number of epochs
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

### Usage
# load data   
'''data_processor = DataProcessor('snappfood-data/train.csv', 'snappfood-data/test.csv')
data_processor.process()    

input_train = data_processor.input_train
lable_tensor = data_processor.labels_tensor_train

##




feature_extractor = BERTFeatureExtractor()
features_cls = feature_extractor.extract_features_cls(input_train[0:10])
features_4_layers = feature_extractor.extract_features_4_layers(input_train[0:10])

# Initialize models, loss functions, and optimizers
model_cls = SimpleNN(input_size=768, num_classes=2)
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=0.001)

model_4_layers = SimpleNN(input_size=768 * 4, num_classes=2)  # Adjust input size for concatenated features
criterion_4_layers = nn.CrossEntropyLoss()
optimizer_4_layers = torch.optim.Adam(model_4_layers.parameters(), lr=0.001)

# Train the models
trainer_cls = ModelTrainer(model_cls, criterion_cls, optimizer_cls)
trainer_cls.train(features_cls, lable_tensor[0:10])

trainer_4_layers = ModelTrainer(model_4_layers, criterion_4_layers, optimizer_4_layers)
trainer_4_layers.train(features_4_layers, lable_tensor[0:10])
'''