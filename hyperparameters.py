import numpy as np
import pandas as pd

#%%
vocab_size = 5000
embedding_dim = 128
max_length = 64
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .7
batch_size = 128
# check_random_article = np.random.randint(150)
# print("Random checking of articles in index : " + str(check_random_article))
## Model specs

node = 256 # number of nodes
nLayers = 6 # number of  hidden layer
dropout = 0.5
nClasses = 77
num_epochs = 1000
#%%

# TRAINING_URL = "https://raw.githubusercontent.com/ardimirzaei/HISB_Mapping_Databases/master/TRAINING_RANGES.csv"
# TRAINING_RANGES = pd.read_csv(TRAINING_URL) # Training URL in hyperparameter file
# TRAINING_RANGES = np.asarray(TRAINING_RANGES[TRAINING_RANGES['include']==1].iloc[:,0])



# PREDICTION_URL = "https://raw.githubusercontent.com/ardimirzaei/HISB_Mapping_Databases/master/PREDICTION_RANGES.csv"
# PREDICTION_RANGES = pd.read_csv(PREDICTION_URL) # PREDICTION URL in hyperparameter file
# PREDICTION_RANGES = np.asarray(PREDICTION_RANGES[PREDICTION_RANGES['include']==1].iloc[:,0])

print("\n")
print("For LSTM the following hyperparameters are;")
print("\n")
print(" - Vocabulary Size : " + str(vocab_size))
print(" - Embedding Dimensions : " + str(embedding_dim))
print(" - Maximum Length of Each Sequence : " + str(max_length))
print(" - Truncation Type : " + str(trunc_type))
print(" - Padding Type : " + str(padding_type))
print(" - Out Of Vocabulary (OOV) Token : " + str(oov_tok))
print(" - Training Portion   " + str(training_portion))
print(" - Batch Size : " + str(batch_size))
print("-"*25)
# print("Random checking of articles in index : " + str(check_random_article))
print("-"*25)
print("\n")
print("Neural Net Hyperparameters;")
print("\n")
print(" - Number of Nodes Per Layer : " + str(node))
print(" - Number of Fully Connected Hidden Layers : " + str(nLayers))
print(" - Base Dropout Rate : " + str(dropout))
print(" - Maximum Number of Classes to Predict : " + str(nClasses))
print(" - Maximum Number of Epochs : " + str(num_epochs))
print("-"*25)
print("\n")
# print("Training Datasets : " + str(TRAINING_RANGES))
# print("\n")
# print("Prediction Datasets : " + str(PREDICTION_RANGES))