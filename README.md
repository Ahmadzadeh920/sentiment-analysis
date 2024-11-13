# Sentiment Analysis of User Reviews using BERT
 In this project, you will perform a set of sentiment analysis using BERT (Bidirectional Encoder Representations of Transformers). You will explore the construction and different methods of using BERT for text classification. You work with the SnappFood dataset will contain 70,000 user comments with positive or negative tags.


This exercise is divided into several parts:
1. Data preprocessing
2. Using pre-trained BERT for feature extraction
3. Tuning-Fine (the last layer of BERT)
4. Full fine adjustment of BERT

## Data Preprocessing Module
- **Purpose**: Preparing the SnappFood dataset to enter the BERT model
- **Duties**:
  1. Loading the dataset
  2. Cleaning data
  3. Encoding labels
  4. Convert text labels ("positive", "negative") to numerical labels (1 for positive, 0 for negative).
  5. Tokenization
     - Tokenize comments using the BERT tokenizer.
  6. Create DataLoaders

### Usage
```bash
from preproces import DataProcessor
data_processor = DataProcessor('snappfood/train.csv', 'snappfood/test.csv')
data_processor.process()      
train_dataloader = data_processor.dataloader_train
test_dataloader = data_processor.dataloader_tst

```

# Extractor module 
- **Purpose**
Using BERT as a feature extractor and comparing different feature extraction methods for classification.

- **Duties**:

1. Load the pretrained BERT
2. Feature extraction using the [CLS] vector
3. Feature extraction by adding word embeddings in the last 4 hidden layers
4. Creating categories
5. Training
o Train each classifier on the training data.
6. Evaluation
o Evaluate categories on test data and criteria such as accuracy, positive precision, recall
 (recall and report 1F score)

### Usage:
```bash
import importlib
Extractor_module = importlib.import_module('Extractor')

importlib.reload(Extractor_module)
from Extractor import BERTFeatureExtractor,SimpleNN, ModelTrainer
import torch
import torch.nn as nn
lable_tensor = data_processor.labels_tensor_train
input_train = data_processor.input_train
feature_extractor = BERTFeatureExtractor()
features_cls = feature_extractor.extract_features_cls(input_train)
features_4_layers = feature_extractor.extract_features_4_layers(input_train)

# Initialize models, loss functions, and optimizers
model_cls = SimpleNN(input_size=768, num_classes=2)
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=0.001)


model_4_layers = SimpleNN(input_size=768 , num_classes=2)  # Adjust input size for concatenated features
criterion_4_layers = nn.CrossEntropyLoss()
optimizer_4_layers = torch.optim.Adam(model_4_layers.parameters(), lr=0.001)

# Train the models
trainer_cls = ModelTrainer(model_cls, criterion_cls, optimizer_cls)
trainer_cls.train(features_cls, lable_tensor)

trainer_4_layers = ModelTrainer(model_4_layers, criterion_4_layers, optimizer_4_layers)
trainer_4_layers.train(features_4_layers, lable_tensor)	
```

# Set_final_layer Module
- **Purpose** Fine-tune only the last layer of BERT along with the handle.
- **Duties**:
1. Changing the BERT model(Freeze all BERT layers except the last encoder layer)
2. Adding a classification layer
3. Adjust the optimizer
4. Training
5. Evaluation(Evaluate the tuned model on test data)
### Usage
```bash
import importlib
Set_final_layer_module = importlib.import_module('Set_final_layer_Bert')
from Set_final_layer_Bert import BERTClassifier
importlib.reload(Set_final_layer_module)
# Create an instance of BERTClassifier
bert_classifier = Set_final_layer_module.BERTClassifier()
# Train the model
bert_classifier.train(train_dataloader, num_epochs=3)
bert_classifier.evaluate(test_dataloader)

```
# Set_all_layers_BERT Module
- **Purpose**
Fine-tune all layers of BERT for the sentiment analysis task.
Full fine-tuning of BERT allows the model to analyze all layers to better adapt to the specific details of the analysis task. 

- **Duties**:
1. Preparing the BERT model
2. Setting the learning rate optimizer and scheduler
3. Training
( Train the model on the training data).
4. Evaluation

### Usage
```bash
import importlib
Set_all_layer_module = importlib.import_module('set_all_layers_Bert')
from set_all_layers_Bert import Set_all_layers
importlib.reload(Set_all_layer_module)
# Create an instance of BERTSentimentAnalyzer
Set_all_layers_instance = Set_all_layer_module.Set_all_layers()
# Train the model
Set_all_layers_instance.train(train_dataloader)
Set_all_layers_instance.evaluate(test_dataloader)
```




