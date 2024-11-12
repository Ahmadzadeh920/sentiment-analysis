import torch
from transformers import BertForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

class Set_all_layers:
    
    # initializes the model and tokenizer, does not freeze any layers, and sets up the optimizer and scheduler.
    def __init__(self, model_name='bert-base-multilingual-cased', num_labels=2, learning_rate=5e-5, num_epochs=3):
        # Load the BERT model and tokenizer
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set up the optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Set up the learning rate scheduler
        self.num_epochs = num_epochs
        self.total_steps = None  # This will be set during training
        self.scheduler = None
        
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
            
            if self.scheduler:
                self.scheduler.step()  # Update the learning rate schedule
        
        return total_loss / len(data_loader)

    # orchestrates the training process over a specified number of epochs.
    def train(self, data_loader):
        self.total_steps = len(data_loader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,  # You can change this as needed
            num_training_steps=self.total_steps
        )
        
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(data_loader)
            print(f'Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}')
            
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
