import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer

class DataProcessor:
    # The constructor initializes file paths and the tokenizer
    def __init__(self, train_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.input_train = []
        self.output_train = []
        self.input_tst = []
        self.output_tst = []
        self.dataloader_train = None
        self.dataloader_tst = None
        self.labels_tensor_train = []
        self.labels_tensor_tst = []
    
    # Reads the CSV files.
    def read_data(self):
        # Read the dataset from the CSV files
        data_train = pd.read_csv(self.train_file_path,  sep='; , .')
        data_tst = pd.read_csv(self.test_file_path,  sep='; , .')
        return data_train, data_tst
    
    
    # Cleans and prepares the data.
    # data_train , data_tst are outputs of read_data
    def preprocess_data(self, data_train, data_tst):
        # Remove empty and duplicate input
        df_remove_train = data_train.dropna().drop_duplicates()
        df_remove_tst = data_tst.dropna().drop_duplicates()
        

        # Create input and output matrix
        input_train = [[] for i in range(len(df_remove_train))]    
        output_train = [0 for i in range(len(df_remove_train))]

        for i in range(0, len(df_remove_train)):
            spilt = df_remove_train[i:i+1].to_string().split('\\')
            input_train[i] = spilt[3]
            output_train[i] = spilt[5][1]
    
    
        input_tst = [[] for i in range(len(df_remove_tst))]
               
        output_tst = [0 for i in range(len(df_remove_tst))]

        for j in range(0, len(df_remove_tst)):
            spilt_tst = df_remove_tst[j:j+1].to_string().split('\\')
            input_tst[j] = spilt_tst[3]
            output_tst[j] = spilt_tst[5][1]
            
        
        
        # Create new lists that filter out the unwanted elements
        filtered_input_train = [input_train[i] for i in range(len(input_train)) if input_train[i] != '' and output_train[i] != 'S']
        filtered_output_train = [output_train[i] for i in range(len(output_train)) if input_train[i] != '' and output_train[i] != 'S']

        # Update the original lists
        self.input_train = filtered_input_train
        self.output_train = filtered_output_train


        # Test the input and output
        filtered_input_tst = [input_tst[i] for i in range(len(input_tst)) if input_tst[i] != '' and output_tst[i] != 'S']
        filtered_output_tst = [output_tst[i] for i in range(len(input_tst)) if input_tst[i] != '' and output_tst[i] != 'S']

        # Update the original lists
        self.input_tst = filtered_input_tst
        self.output_tst = filtered_output_tst  
        self.output_train = [int(x) for x in self.output_train if x.isdigit()]
        self.output_tst = [int(x) for x in self.output_tst if x.isdigit()]

            
        
        
    def tokenize_data(self):
        # Tokenize input data
        encoded_comments_train = self.tokenizer(
            self.input_train,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        encoded_comments_tst = self.tokenizer(
            self.input_tst,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )

        return encoded_comments_train, encoded_comments_tst


    # encode_comments_train and encode_comments_tst are the outputs of tokenize_data
    def create_dataloader(self, encoded_comments_train, encoded_comments_tst):
        # Create DataLoader for training data
        input_ids_train = encoded_comments_train['input_ids']
        
        attention_mask_train = encoded_comments_train['attention_mask']
        self.output_train.append(0)
        self.labels_tensor_train = torch.tensor(self.output_train)

        dataset_train = TensorDataset(input_ids_train, attention_mask_train, self.labels_tensor_train)
        self.dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=True)

        # Create DataLoader for test data
        input_ids_tst = encoded_comments_tst['input_ids']
        attention_mask_tst = encoded_comments_tst['attention_mask']
        self.labels_tensor_tst = torch.tensor(self.output_tst)

        dataset_tst = TensorDataset(input_ids_tst, attention_mask_tst, self.labels_tensor_tst)
        self.dataloader_tst = DataLoader(dataset_tst, batch_size=2, shuffle=True)

    def process(self):
        data_train, data_tst = self.read_data()
        self.preprocess_data(data_train, data_tst)
        encoded_comments_train, encoded_comments_tst = self.tokenize_data()
        self.create_dataloader(encoded_comments_train, encoded_comments_tst)

# Usage
'''data_processor = DataProcessor('snappfood-data/train.csv', 'snappfood-data/test.csv')
data_processor.process()'''