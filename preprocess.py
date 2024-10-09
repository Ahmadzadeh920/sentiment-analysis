
# %% [markdown]
# ## read data from Excel file

# %%
import pandas as pd

# Specify the path to your Excel file
file_path_train = 'snappfood-data/train.csv'
file_path_tst = 'snappfood-data/test.csv'


# Read the dataset from the Excel file
data_train = pd.read_csv(file_path_train ,  sep='; , .')
data_tst = pd.read_csv(file_path_tst, sep='; , .')

# Display the first few rows of the dataframe
print(data_train.head())

# %% [markdown]
# ## remove empty and duplicate input

# %%
df_remove_train = data_train.dropna().drop_duplicates()
df_remove_tst = data_tst.dropna().drop_duplicates()


# %%
len(df_remove_train)

# %% [markdown]
# ## create input and output matrix 

# %%
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
    

# %% [markdown]
# ## check the values of input and output

# %%
# Create new lists that filter out the unwanted elements
filtered_input_train = [input_train[i] for i in range(len(input_train)) if input_train[i] != '' and output_train[i] != 'S']
filtered_output_train = [output_train[i] for i in range(len(output_train)) if input_train[i] != '' and output_train[i] != 'S']

# Update the original lists
input_train = filtered_input_train
output_train = filtered_output_train


# Test the input and output
filtered_input_tst = [input_tst[i] for i in range(len(input_tst)) if input_tst[i] != '' and output_tst[i] != 'S']
filtered_output_tst = [output_tst[i] for i in range(len(input_tst)) if input_tst[i] != '' and output_tst[i] != 'S']

# Update the original lists
input_tst = filtered_input_tst
output_tst = filtered_output_tst
        


    

# %%
output_train = [int(x) for x in output_train if x.isdigit()]
output_tst = [int(x) for x in output_tst if x.isdigit()]




# %%
print(output_train[0:10])

# %% [markdown]
# ## tokenize input_train and tst

# %%
from transformers import BertTokenizer
tokenizer_train = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# Tokenize with padding and truncation
encoded_comments_trian = tokenizer_train(
    input_train,
    padding=True,           # Pads to the longest sequence in the batch
    truncation=True,        # Truncates any input longer than max_length
    max_length=64,         # Adjust max_length based on your requirements
    return_tensors='pt'    # Return PyTorch tensors (use 'tf' for TensorFlow)
)



# %%
tokenizer_tst = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# Tokenize with padding and truncation
encoded_comments_tst = tokenizer_tst(
    input_tst,
    padding=True,           # Pads to the longest sequence in the batch
    truncation=True,        # Truncates any input longer than max_length
    max_length=64,         # Adjust max_length based on your requirements
    return_tensors='pt'    # Return PyTorch tensors (use 'tf' for TensorFlow)
)



# %%
output_train.append(0)

# %% [markdown]
# # Data Loader

# %%
import torch
from torch.utils.data import TensorDataset, DataLoader



# and labels are the corresponding labels for each input.
# Assume input_ids and attention_mask are the tokenized input tensors, 
input_ids_train = encoded_comments_trian['input_ids']  # Example tokenized input
attention_mask_train = encoded_comments_trian['attention_mask']  # Attention masks
 

# Convert lists to PyTorch tensors
input_ids_tensor = torch.tensor(input_ids_train)
attention_mask_tensor = torch.tensor(attention_mask_train)
labels_tensor = torch.tensor(output_train)

# Create a TensorDataset
dataset_train = TensorDataset(input_ids_tensor, attention_mask_tensor, labels_tensor)

# Create DataLoader
dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=True)




# %%
# and labels are the corresponding labels for each input.
# Assume input_ids and attention_mask are the tokenized input tensors, 
input_ids_tst = encoded_comments_tst['input_ids']  # Example tokenized input
attention_mask_tst = encoded_comments_tst['attention_mask']  # Attention masks
 

# Convert lists to PyTorch tensors
input_ids_tensor = torch.tensor(input_ids_tst)
attention_mask_tensor = torch.tensor(attention_mask_tst)
labels_tensor = torch.tensor(output_tst)

# Create a TensorDataset
dataset_tst = TensorDataset(input_ids_tensor, attention_mask_tensor, labels_tensor)

# Create DataLoader
dataloader_tst = DataLoader(dataset_tst, batch_size=2, shuffle=True)





