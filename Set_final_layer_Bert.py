import preproces as pre
import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader

class BERTClassifier:
    
    # initializes the model and tokenizer, freezes the appropriate layers, and sets up the optimizer.
    def __init__(self, model_name='bert-base-multilingual-cased', num_labels=2, learning_rate=5e-5):
        # Load the BERT model and tokenizer
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze all BERT layers except the last encoder layer
        for name, param in self.model.bert.named_parameters():
            if "encoder.layer.11." not in name:  # Keep the last encoder layer trainable
                param.requires_grad = False
        
        # Set up the optimizer
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
    # method handles the training for one epoch, processing batches from the data loader and updating the model weights.
    def train_epoch(self, data_loader):
        self.model.train()  # Set the model to training mode
        total_loss = 0
        
        for batch in data_loader:
            input_ids = batch[0]
            attention_mask = batch[1]
            labels = batch[2]
            
            self.optimizer.zero_grad()  # Clear previous gradients
            
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()  # Backpropagation
            self.optimizer.step()  # Update weights
        
        return total_loss / len(data_loader)
    # orchestrates the training process over a specified number of epochs.

    def train(self,data_loader, num_epochs=3):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(data_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}')
            
            
            
    # Method to evaluate the model on test data
    def evaluate(self, data_loader):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():  # Disable gradient calculation
            for batch in data_loader:
                input_ids = batch[0]
                attention_mask = batch[1]
                labels = batch[2]

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                # Calculate correct predictions
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_predictions

        print(f'Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
        return avg_loss, accuracy


### Usage
'''bert_classifier = BERTClassifier()
# load data   
data_processor = DataProcessor('snappfood-data/train.csv', 'snappfood-data/test.csv')
data_processor.process()    
train_dataloader = data_processor.dataloader_train,
test_dataloader = data_processor.dataloader_tst
##
bert_classifier.train(train_dataloader,num_epochs=3)
bert_classifier.evaluate(test_dataloader)'''