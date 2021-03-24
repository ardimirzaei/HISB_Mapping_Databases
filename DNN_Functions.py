
from keras.layers import  Dropout, Dense, LSTM, Embedding, Bidirectional, Activation, Flatten,Input, concatenate

from keras.models import Sequential, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers.core import Activation

from sklearn.model_selection import StratifiedShuffleSplit
from Grab_gSheets_Functions import get_google_sheet, gsheet_to_df
import pandas as pd
from datetime import datetime, timedelta


import  matplotlib.pyplot as plt
from hyperparameters import *


##########
# Glove commands
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, classification_report, log_loss, plot_confusion_matrix, accuracy_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Flatten
from keras.layers import Dropout, Conv1D, GlobalMaxPool1D, GRU, GlobalAvgPool1D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
# from contractions import contractions

#####


# from gather_explore_prepare import *

def Bag_of_Words_DNN(shape, nClasses, dropout=dropout, node = node, nLayers = nLayers):
    """
    buildModel_DNN_Tex(shape, nClasses,dropout)
    Build Deep neural networks Model for text classification using Bag of Words methods. 
    Shape is input feature space (shape = train_BoW.shape[1])
    nClasses is number of classes
    """
    model = Sequential()
    # node = 512 # number of nodes
    # nLayers = 6 # number of  hidden layer
    model.add(Dense(node,input_dim=shape,activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        model.add(Dense(node,input_dim=node,activation='tanh'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def LSTM_DNN(shape, nClasses, dropout=dropout, node = node, nLayers = nLayers, vocab_size = vocab_size, embedding_dim = embedding_dim, max_length = max_length):
    """



    """
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim,input_length=max_length))
    for i in range(0,1):
        model.add(Bidirectional(LSTM(embedding_dim, activation='tanh', return_sequences=True)))
        model.add(Dropout(dropout))
        
    model.add(Bidirectional(LSTM(embedding_dim, activation='tanh', return_sequences=False)))
    model.add(Dropout(dropout))

    model.add(Dense(node,input_dim=embedding_dim,activation='tanh'))
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        model.add(Dense(node,input_dim=node,activation='tanh'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def Combo_BoW_LSTM_DNN(shape1, shape2, nClasses = nClasses, dropout=dropout, node = node, nLayers = nLayers, vocab_size = vocab_size, embedding_dim = embedding_dim, max_length = max_length):
    """
    shape1: size of the BoW train data. Example is X_train_BoW.shape[1]
    shape2: size of the padded sequences. This is usually max_length which is equal to 64
    """

    # define two sets of inputs
    inputA = Input(shape=(shape1,))
    inputB = Input(shape=(shape2,))
    # the first branch operates on the first input
    x = Dense(node, activation="relu")(inputA)
    x = Dropout(dropout, )(x)
    for i in range(0,nLayers):
        x = Dense(node, activation="relu")(x)
        x = Dropout(dropout, )(x)
    x = Dense(nClasses, activation='softmax')(x)

    x = Model(inputs=inputA, outputs=x)

    # the second branch opreates on the second input
    y = Embedding(vocab_size, embedding_dim,input_length=max_length)(inputB)
    y = Bidirectional(LSTM(embedding_dim, activation='tanh', return_sequences=True))(y)
    y = Dropout(dropout)(y)
    #for i in range(0,1):
    #    y = Bidirectional(LSTM(embedding_dim, activation='tanh', return_sequences=True))(y)
    #    y = Dropout(dropout)(y)
    y = Bidirectional(LSTM(embedding_dim, activation='tanh', return_sequences=False))(y)
    y = Dropout(dropout)(y)

    y = Dense(node, activation="relu")(y)
    y = Dropout(dropout, )(y)
    for i in range(0,nLayers):
        y = Dense(node, activation="relu")(y)
        y = Dropout(dropout, )(y)

    y = Dense(nClasses, activation='softmax')(y)

    y = Model(inputs=inputB, outputs=y)
    # combine the output of the two branches
    combined = concatenate([x.output, y.output])
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(nClasses, activation="softmax")(combined)
    #z = Dense(1, activation="linear")(z)
    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[x.input, y.input], outputs=z)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model






#%%
def plot_metrics(model_DNN, model="Unnamed", expStyle = False, EXPERIMENT_NAME = "Unnamed"):
    """
    Model is used for saving the name. Choose LSTM or BoW
    """
    EXPERIMENT_NAME = EXPERIMENT_NAME

    plt.figure(figsize=(10,6))  
    plt.plot(model_DNN.history.history['acc'])
    plt.plot(model_DNN.history.history['val_acc'])
    plt.plot(model_DNN.history.history['loss'])
    plt.plot(model_DNN.history.history['val_loss'])
    plt.legend(['train accuracy','test accuracy','train loss','test loss'],loc='upper right')
    plt.show()
    if expStyle:
        plt.savefig(str(EXPERIMENT_NAME) + str(datetime.now().strftime("%Y%m%d-%H%M%S")) + '_metrics_' + str(model))
    else: 
        plt.savefig(str(datetime.now().strftime("%Y%m%d-%H%M%S")) + '_metrics_' + str(model))



#%%
################################
# GLOVE EMBEDDINGS

def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

def get_embdedings_matrix(embeddings_index, word_index, nb_words = None):
    all_embs = np.stack(embeddings_index.values())
    print('Shape of Full Embeddding Matrix', all_embs.shape)
    embed_dims = all_embs.shape[1]
    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    #best to free up memory, given the size, which is usually ~3-4GB in memory
    del all_embs
    if nb_words is None:
        nb_words = len(word_index)
    else:
        nb_words = min(nb_words, len(word_index))
    # If the plus works, it fixes the issue of dimension indicies being out of range
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words +1, embed_dims)) 
    found_vectors = 0
    words_not_found = []
    for word, i in tqdm(word_index.items()):
        if i >= nb_words: 
            continue
        embedding_vector = None
        if word in embeddings_index:
            embedding_vector = embeddings_index.get(word)
        elif word.lower() in embeddings_index:
            embedding_vector = embeddings_index.get(word.lower())
        # for twitter check if the key is a hashtag
        elif '#'+word.lower() in embeddings_index:
            embedding_vector = embeddings_index.get('#'+word.lower())
            
        if embedding_vector is not None: 
            found_vectors += 1
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append((word, i))

    print("% of Vectors found in Corpus", found_vectors / nb_words)
    return embedding_matrix, words_not_found

def load_glove(word_index):
#     print('Loading Glove')
    embed_file_path = 'input/glove.840B.300d.txt'
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in tqdm(open(embed_file_path, encoding="utf-8")))
    print("Built Embedding Index:", len(embeddings_index))
    return get_embdedings_matrix(embeddings_index, word_index)




def GLV_LSTM(embed_matrix, nClasses, dropout=dropout, node = node, nLayers = nLayers, vocab_size = vocab_size, embedding_dim = embedding_dim, max_length = max_length):
    """
    Extends Model 5 and adds another Bidirectional LSTM layer
    """
    model = Sequential()
    model.add(Embedding(input_dim = embed_matrix.shape[0], output_dim = embed_matrix.shape[1], input_length = max_length,  weights=[embed_matrix], trainable=True))
    # model.add(Embedding(input_dim = vocab_size, output_dim = embed_matrix.shape[1], input_length = max_length,  weights=[embed_matrix], trainable=True))
    model.add(SpatialDropout1D(0.3))
    model.add(Bidirectional(LSTM(128, dropout=dropout, recurrent_dropout=dropout, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=dropout, recurrent_dropout=dropout, return_sequences=True)))
    model.add(Conv1D(512, 4))
    model.add(GlobalMaxPool1D())
    model.add(Dense(node,input_dim=512,activation='tanh'))
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        model.add(Dense(node,input_dim=node,activation='tanh'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model



def GLV_FC(embed_matrix, nClasses, dropout=dropout, node = node, nLayers = nLayers, vocab_size = vocab_size, embedding_dim = embedding_dim, max_length = max_length):
    """
    Extends Model 5 and adds another Bidirectional LSTM layer
    """
    model = Sequential()
    model.add(Embedding(input_dim = embed_matrix.shape[0], output_dim = embed_matrix.shape[1], input_length = max_length,  weights=[embed_matrix], trainable=True))
    # model.add(Embedding(input_dim = vocab_size, output_dim = embed_matrix.shape[1], input_length = max_length,  weights=[embed_matrix], trainable=True))
    model.add(SpatialDropout1D(0.3))
    model.add(Conv1D(512, 4))
    model.add(GlobalMaxPool1D())
    model.add(Dense(node,input_dim=512,activation='tanh'))
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        model.add(Dense(node,input_dim=node,activation='tanh'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

















































# def decode_article_BoW(text):
#     text = np.argwhere(text>0).ravel()
#     return ' '.join([reverse_word_index.get(i, '?') for i in text])

# def decode_article_LSTM(text):
#     return ' '.join([reverse_word_index.get(i, '?') for i in text])




# #def gather_prepare_load(x_Data, y_Data):
#     TRAINING_RANGES = x_Data
#     PREDICTION_RANGES = y_Data
#     '''
#     1. Gather Data - DOWNLOAD GOOGLE SHEET
#     '''

#     print("-"*25)
#     print("\n")
#     print("Training Datasets : " + str(TRAINING_RANGES))
#     print("\n")
#     print("Prediction Datasets : " + str(PREDICTION_RANGES))


#     SPREADSHEET_ID = '1j9BHMXVBOgEOwQtacEMLG-YWSasy_1b04LK-8T_uJ8Y'

#     # LOAD TRAINING DATA

#     if len(TRAINING_RANGES)>1:
#         TrainingData =  gsheet_to_df(get_google_sheet(SPREADSHEET_ID, str(TRAINING_RANGES[0]+'!A:C')))
#         for RANGE_NAME in TRAINING_RANGES[1:]:
#             TrainingData =  pd.concat([TrainingData, gsheet_to_df(get_google_sheet(SPREADSHEET_ID, str(RANGE_NAME+'!A:C')))])
#     else:
#         TrainingData =  gsheet_to_df(get_google_sheet(SPREADSHEET_ID, str(TRAINING_RANGES[0]+'!A:C')))

#     # LOAD DATA TO PREDICT
#     if len(PREDICTION_RANGES)>1:
#         PredictionData =  gsheet_to_df(get_google_sheet(SPREADSHEET_ID, str(PREDICTION_RANGES[0]+'!A:C')))
#         for RANGE_NAME in PREDICTION_RANGES[1:]:
#             PredictionData =  pd.concat([PredictionData, gsheet_to_df(get_google_sheet(SPREADSHEET_ID, str(RANGE_NAME+'!A:C')))])
#     else:
#         PredictionData =  gsheet_to_df(get_google_sheet(SPREADSHEET_ID, str(PREDICTION_RANGES[0]+'!A:C')))

#     #%%
#     '''
#     2. Explore - Duplicate rows with only 1 count
#     '''
#     MergeData = TrainingData.reset_index().drop(columns='index') 
#     LowCountRows = MergeData.groupby('Coded').filter(lambda x: len(x)<2).index.tolist()

#     MergeData = pd.concat([MergeData.iloc[LowCountRows,],MergeData]).reset_index().drop(columns='index') # Add the rows on top that are low count

#     articles = MergeData['Label']
#     labels =  MergeData['Coded']

#     #%%

#     '''
#     3. Prepare Data - 
#     '''
#     # DO I NEED THIS LITTLE BIT HERE
#     ##########
#     # train_size = int(len(articles) * training_portion)

#     # train_articles = articles[0: train_size]
#     # train_labels = labels[0: train_size]

#     # validation_articles = articles[train_size:]
#     # validation_labels = labels[train_size:]
#     ##########



#     sss = StratifiedShuffleSplit(n_splits=1, test_size=1-training_portion, random_state=0) #change from 0.1 to 0.3 after HINTS

#     for train_index, test_index in sss.split(X=articles, y= labels):
#         print("TRAIN:", train_index, "TEST:", test_index)
#         train_articles, validation_articles = articles[train_index], articles[test_index] 
#         train_labels, validation_labels = labels[train_index], labels[test_index]


#     # print("Sizes of the training and validation articles and labels" + str(train_size))
#     print("Training X set: " + str(len(train_articles)))
#     print("Training Y set: " + str(len(train_labels)))
#     print("Validation X set: " + str(len(validation_articles)))
#     print("Validation X set: " + str(len(validation_labels)))

#     #%%
#     '''
#     Tokenzier
#     '''

#     tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
#     tokenizer.fit_on_texts(articles)
#     word_index = tokenizer.word_index


#     '''
#     Lokenizing the Labels
#     '''

#     label_tokenizer = Tokenizer(
#         num_words=None, filters='#', lower=False,
#         split='#', char_level=False, oov_token=oov_tok, document_count=0)

#     label_tokenizer.fit_on_texts(labels)

#     training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))

#     validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))


#     #%%

#     '''
#     Article Decoding
#     '''

#     reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#     return SPREADSHEET_ID, TrainingData, PredictionData, MergeData, LowCountRows, articles, labels, train_articles, train_labels, validation_articles, validation_labels, tokenizer, word_index, label_tokenizer, training_label_seq, validation_label_seq, reverse_word_index


# # def model_data_preperation(model, train_articles, validation_articles):
#     """
#     Choose either model = 
#     "LSTM"   or
#     "BoW"
#     # """
#     # tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    
#     check_random_article = np.random.randint(150)
#     print("Random checking of articles in index : " + str(check_random_article))
    
#     if model == "LSTM":
#         print("Sequencing LSTM")
#         print("-"*25)
#         train_sequences = tokenizer.texts_to_sequences(train_articles)
#         train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#         print(train_sequences[check_random_article])
#         print("Length of the training sequence : " + str(len(train_sequences[0])))
#         print("Length of the training sequence with padding: " + str(len(train_padded[0])))

#         validation_sequences = tokenizer.texts_to_sequences(validation_articles)
#         validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#         print("Length of the validation sequence : " + str(len(validation_sequences[0])))
#         print("Length of the validation sequence with padding: " + str(len(validation_padded[0])))

#         print("-"*25)
#         print("Decoding LSTM Sequence")
#         print(decode_article_LSTM(train_padded[check_random_article]))
#         print(np.array(train_articles)[check_random_article]) 

#         print("Returning train_padded, validation_padded")
#         print('-'*25)

#         return train_padded, validation_padded
#     elif model == "BoW":
#         print("Bag of Words Model")
#         print("-"*25)

#         train_BoW = tokenizer.texts_to_matrix(train_articles)

#         print(np.argwhere(train_BoW[check_random_article]>0))
#         print("Bag of Words Training shape = " + str(train_BoW.shape))

#         validation_BoW = tokenizer.texts_to_matrix(validation_articles)

#         print("Bag of Words Validation shape = " + str(validation_BoW.shape))

#         print("-"*25)
#         print("Decoding BoW")
#         print(decode_article_BoW(train_BoW[check_random_article]))
#         print(np.array(train_articles)[check_random_article]) 

#         print("Returning train_train_BoW, validation_BoW")
#         print('-'*25)
#         return train_BoW, validation_BoW
#     elif model == "Combo_BoWLSTM": 
#         print("Sequencing LSTM")
#         print("-"*25)
#         train_sequences = tokenizer.texts_to_sequences(train_articles)
#         train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#         print(train_sequences[check_random_article])
#         print("Length of the training sequence : " + str(len(train_sequences[0])))
#         print("Length of the training sequence with padding: " + str(len(train_padded[0])))

#         validation_sequences = tokenizer.texts_to_sequences(validation_articles)
#         validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#         print("Length of the validation sequence : " + str(len(validation_sequences[0])))
#         print("Length of the validation sequence with padding: " + str(len(validation_padded[0])))

#         print("-"*25)
#         print("Decoding LSTM Sequence")
#         print(decode_article_LSTM(train_padded[check_random_article]))
#         print(np.array(train_articles)[check_random_article]) 


#         print('-'*25)
        
#         print("Bag of Words Model")
#         print("-"*25)

#         train_BoW = tokenizer.texts_to_matrix(train_articles)

#         print(np.argwhere(train_BoW[check_random_article]>0))
#         print("Bag of Words Training shape = " + str(train_BoW.shape))

#         validation_BoW = tokenizer.texts_to_matrix(validation_articles)

#         print("Bag of Words Validation shape = " + str(validation_BoW.shape))

#         print("-"*25)
#         print("Decoding BoW")
#         print(decode_article_BoW(train_BoW[check_random_article]))
#         print(np.array(train_articles)[check_random_article]) 

#         print("Returning train_train_BoW, validation_BoW")        
#         print("Returning train_padded, validation_padded")
#         print('-'*25)


#         return  train_BoW, validation_BoW, train_padded, validation_padded


#     else: 
#         print("No model choosen, cannot do sequencing")

#     if not model :
#         # print("No model choosen, cannot do sequencing")



