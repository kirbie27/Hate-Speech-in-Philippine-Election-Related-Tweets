#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load models and necessary files


# In[2]:


#Imports for the models
#import necessary libraries
import pandas as pd
import csv
import re
import validators
import emoji
import unidecode
import nltk
import pickle
nltk.download('stopwords')
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import WhitespaceTokenizer
from itertools import chain
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import keras
from keras.preprocessing import sequence
import tensorflow as tf
from keras import Input
from keras import optimizers
from keras import backend as K
import torch
from keras import regularizers
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_blobs
from keras.models import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D 
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import pyplot
from keras.models import load_model
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, f_classif

import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D 
from keras.utils import np_utils
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from itertools import chain

from tqdm import tqdm
from gensim.models import fasttext
from gensim.test.utils import datapath
import os, re, csv, math, codecs, pickle, nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[3]:


#Initializing necessary variables,lists,objects, and functions
#Define necessary functions


#Functions for the gui

def classify():
    start_time = time.time()
    user_input = inputtxt.get('1.0',END)
    print(user_input)
    tokenized_tweet = pre_process_tweet(user_input)
    
    #classification
    
    #binary classifications
    bf_7030.set(fasttextcnn_predict(tokenized_tweet, binary_7030_tokenizer, max_sentence_len_binary_7030, 
                                    binary_7030_fasttext_model, 0))
    bt_7030.set(tfidfffnn_predict(tokenized_tweet, binary_7030_tfidf, binary_7030_selector, binary_7030_tfidf_model, 0))
    
    bf_8020.set(fasttextcnn_predict(tokenized_tweet, binary_8020_tokenizer, max_sentence_len_binary_8020, 
                                    binary_8020_fasttext_model, 0))
    bt_8020.set(tfidfffnn_predict(tokenized_tweet, binary_8020_tfidf, binary_8020_selector, binary_8020_tfidf_model, 0))
    
    bf_9010.set(fasttextcnn_predict(tokenized_tweet, binary_9010_tokenizer, max_sentence_len_binary_9010, 
                                    binary_9010_fasttext_model, 0))
    bt_9010.set(tfidfffnn_predict(tokenized_tweet, binary_9010_tfidf, binary_9010_selector, binary_9010_tfidf_model, 0))
    
    #multilabel classifications
    
    mf_7030m.set(fasttextcnn_predict(tokenized_tweet, multilabel_7030_tokenizer, max_sentence_len_multilabel_7030, 
                                    multilabel_7030_fasttext_model, 1))
    mt_7030m.set(tfidfffnn_predict(tokenized_tweet, multilabel_7030_tfidf, multilabel_7030_selector, 
                                   multilabel_7030_tfidf_model, 1))
    
    mf_8020m.set(fasttextcnn_predict(tokenized_tweet, multilabel_8020_tokenizer, max_sentence_len_multilabel_8020, 
                                    multilabel_8020_fasttext_model, 1))
    mt_8020m.set(tfidfffnn_predict(tokenized_tweet, multilabel_8020_tfidf, multilabel_8020_selector, 
                                   multilabel_8020_tfidf_model, 1))
    
    mf_9010m.set(fasttextcnn_predict(tokenized_tweet, multilabel_9010_tokenizer, max_sentence_len_multilabel_9010, 
                                    multilabel_9010_fasttext_model, 1))
    mt_9010m.set(tfidfffnn_predict(tokenized_tweet, multilabel_9010_tfidf, multilabel_9010_selector, 
                                   multilabel_9010_tfidf_model, 1))
    
    run_time =  time.time() - start_time
    print("Runtime: ",run_time)
    
def clear():
    inputtxt.delete('1.0', END)
    step_1.set('')
    step_2.set('')
    step_3.set('')
    step_4.set('')
    step_5.set('')
    step_6.set('')
    step_7.set('')
    
    bf_7030.set('')
    bt_7030.set('')
    bf_8020.set('')
    bt_8020.set('')
    bf_9010.set('')
    bt_9010.set('')
    
    mf_7030m.set('')
    mt_7030m.set('')
    mf_8020m.set('')
    mt_8020m.set('')
    mf_9010m.set('')
    mt_9010m.set('')


# In[4]:


#Pre Processing

#Declare Stop Words

filipino_stopwords = set(
    """
akin
aking
ako
alin
am
amin
aming
ang
ano
anumang
apat
at
atin
ating
ay
bababa
bago
bakit
bawat
bilang
dahil
dalawa
dapat
din
dito
doon
gagawin
gayunman
ginagawa
ginawa
ginawang
gumawa
gusto
habang
hanggang
hindi
huwag
iba
ibaba
ibabaw
ibig
ikaw
ilagay
ilalim
ilan
inyong
isa
isang
itaas
ito
iyo
iyon
iyong
ka
kahit
kailangan
kailanman
kami
kanila
kanilang
kanino
kanya
kanyang
kapag
kapwa
karamihan
katiyakan
katulad
kaya
kaysa
ko
kong
kulang
kumuha
kung
laban
lahat
lamang
likod
lima
maaari
maaaring
maging
mahusay
makita
marami
marapat
masyado
may
mayroon
mga
minsan
mismo
mula
muli
na
nabanggit
naging
nagkaroon
nais
nakita
namin
napaka
narito
nasaan
ng
ngayon
ni
nila
nilang
nito
niya
niyang
noon
o
pa
paano
pababa
paggawa
pagitan
pagkakaroon
pagkatapos
palabas
pamamagitan
panahon
pangalawa
para
paraan
pareho
pataas
pero
pumunta
pumupunta
sa
saan
sabi
sabihin
sarili
sila
sino
siya
tatlo
tayo
tulad
tungkol
una
walang
""".split()
)

from nltk.corpus import stopwords
english_stopwords = stopwords.words('english')

search = "leni robredo bongbong marcos isko moreno domagoso manny pacman pacquiao ping lacson ernie abella leody de guzman norberto gonzales jose montemayor jr faisal mangondato"
candidatelist = search.split(" ")

#pre-process tweet input
def pre_process_tweet(tweet_input):
    
    #Step 1 - Extract Tweet from input
    #Tweet = tweet_input
    tweet = tweet_input.strip().replace("\n"," ")
    step_1.set(tweet)
    
    #Step 2 - Data Deidentification
    output = ""
    sentence = tweet.split(" ")
    for part in sentence:
        if not re.match(r"(^|[^@\w])@(\w{1,15})\b", part):
            if len(output) == 0:
                output = f"{part}"
            else:
                output = f"{output} {part}"

    tweets_de_identified = output
    step_2.set(tweets_de_identified)
    
    #Step 3 - URL Removal
    output = ""
    sentence = tweets_de_identified.split(" ")
    for part in sentence:
        valid = validators.url(part)

        if (not valid == True):
            if len(output) == 0:
                output = f"{part}"
            else:
                output = f"{output} {part}"
                
    tweets_url_removed = output
    step_3.set(tweets_url_removed)
    
    #Step 4 - Special Character Processing
    
    emoji_removed = emoji.replace_emoji(tweets_url_removed, replace="[emoji]")
    output = ""
    sentence = emoji_removed.split(" ")
    
    for part in sentence:
        if not (re.match(r"^[_\W]+$", part) or "[emoji]" in part):
            if len(output) == 0:
                output = f"{part}"
            else:
                output = f"{output} {part}"
    
    tweets_specialcharacters_removed = output
    step_4.set(tweets_specialcharacters_removed)
    
    #Step 5 - Normalization, lowercase>removediacritics>remove numerics and symbols>stopwords
    
    #lowercase the text
    lowercased_input = tweets_specialcharacters_removed.lower()

    #remove diacritics
    diacritics_removed = unidecode.unidecode(lowercased_input)

    output = ""
    sentence = diacritics_removed.split(" ")

    for part in sentence:
        part = re.sub("[^A-Za-z ]+$", "", part)
        part = re.sub("^[^A-Za-z #]+", "", part)
        if not (len(part) <= 1 or re.match(r"[^#a-zA-Z]", part) or part in english_stopwords or 
                part in filipino_stopwords or any(part in x for x in candidatelist)):     
            if len(output) == 0:
                output = f"{part}"
            else:
                output = f"{output} {part}"  
                
    tweets_normalized = output
    step_5.set(tweets_normalized)
    
    #Step 6 - Hashtag Processing, removing the hashtags from the tweet
    output = ""
    sentence = tweets_normalized.split(" ")

    for part in sentence:
        if not re.match(r"#(\w+)", part):
            if len(output) == 0:
                output = f"{part}"
            else:
                output = f"{output} {part}"
                
    tweets_hashtags_removed = output  
    step_6.set(tweets_hashtags_removed)
    #Step 7 - Tokenization
    tokenizer = WhitespaceTokenizer()
    
    output = tokenizer.tokenize(tweets_hashtags_removed)
    
    tweets_tokenized = output
    tokens = ','.join(str(s) for s in tweets_tokenized)
    
    step_7.set(tokens)
    
    return tweets_tokenized


# In[5]:


#Load necessary files for models

#Load Models and files

#x
with open('binary/7030/x_train.pkl', 'rb') as file:
    binary_7030_x_train = pickle.load(file)
    
    
with open('binary/8020/x_train.pkl', 'rb') as file:
    binary_8020_x_train = pickle.load(file)

    
with open('binary/9010/x_train.pkl', 'rb') as file:
    binary_9010_x_train = pickle.load(file)
    
    
with open('multilabel/7030/x_train.pkl', 'rb') as file:
    multilabel_7030_x_train = pickle.load(file)
    
    
with open('multilabel/8020/x_train.pkl', 'rb') as file:
    multilabel_8020_x_train = pickle.load(file)
    
    
with open('multilabel/9010/x_train.pkl', 'rb') as file:
    multilabel_9010_x_train = pickle.load(file)
    
#y
with open('binary/7030/y_train.pkl', 'rb') as file:
    binary_7030_y_train = pickle.load(file)
    
#Load Models and files
    
with open('binary/8020/y_train.pkl', 'rb') as file:
    binary_8020_y_train = pickle.load(file)

#Load Models and files
    
with open('binary/9010/y_train.pkl', 'rb') as file:
    binary_9010_y_train = pickle.load(file)
    
#Load Models and files
    
with open('multilabel/7030/y_train.pkl', 'rb') as file:
    multilabel_7030_y_train = pickle.load(file)
    
#Load Models and files
    
with open('multilabel/8020/y_train.pkl', 'rb') as file:
    multilabel_8020_y_train = pickle.load(file)
    
#Load Models and files
    
with open('multilabel/9010/y_train.pkl', 'rb') as file:
    multilabel_9010_y_train = pickle.load(file)
    

    
#for fasttext cnn
binary_7030_fit = list(chain.from_iterable(binary_7030_x_train))
binary_8020_fit = list(chain.from_iterable(binary_8020_x_train))
binary_9010_fit = list(chain.from_iterable(binary_9010_x_train))

multilabel_7030_fit = list(chain.from_iterable(multilabel_7030_x_train))
multilabel_8020_fit = list(chain.from_iterable(multilabel_8020_x_train))
multilabel_9010_fit = list(chain.from_iterable(multilabel_9010_x_train))


list_len = [len(i) for i in binary_7030_x_train]
index_of_max = np.argmax(np.array(list_len))
max_sentence_len_binary_7030 = list_len[index_of_max]

list_len = [len(i) for i in binary_8020_x_train]
index_of_max = np.argmax(np.array(list_len))
max_sentence_len_binary_8020 = list_len[index_of_max]

list_len = [len(i) for i in binary_9010_x_train]
index_of_max = np.argmax(np.array(list_len))
max_sentence_len_binary_9010 = list_len[index_of_max]

list_len = [len(i) for i in multilabel_7030_x_train]
index_of_max = np.argmax(np.array(list_len))
max_sentence_len_multilabel_7030 = list_len[index_of_max]

list_len = [len(i) for i in multilabel_8020_x_train]
index_of_max = np.argmax(np.array(list_len))
max_sentence_len_multilabel_8020 = list_len[index_of_max]

list_len = [len(i) for i in multilabel_9010_x_train]
index_of_max = np.argmax(np.array(list_len))
max_sentence_len_multilabel_9010 = list_len[index_of_max]

#feed train set on vectorizer 
binary_7030_tokenizer = Tokenizer(num_words=100000, char_level=False)
binary_7030_tokenizer.fit_on_texts(binary_7030_fit)

binary_8020_tokenizer = Tokenizer(num_words=100000, char_level=False)
binary_8020_tokenizer.fit_on_texts(binary_8020_fit)

binary_9010_tokenizer = Tokenizer(num_words=100000, char_level=False)
binary_9010_tokenizer.fit_on_texts(binary_9010_fit)

multilabel_7030_tokenizer = Tokenizer(num_words=100000, char_level=False)
multilabel_7030_tokenizer.fit_on_texts(multilabel_7030_fit)

multilabel_8020_tokenizer = Tokenizer(num_words=100000, char_level=False)
multilabel_8020_tokenizer.fit_on_texts(multilabel_8020_fit)

multilabel_9010_tokenizer = Tokenizer(num_words=100000, char_level=False)
multilabel_9010_tokenizer.fit_on_texts(multilabel_9010_fit)


#vectorizer and tfidf for tfidf models

#Convert Labels from Strings to categorical Integers {Non-Hate = 1, Hate = 0}
mapping_binary = {'Non-hate': 0, 'Hate': 1}
mapping_multilabel = {'Positive': 0, 'Negative': 1, 'Neutral':2}

binary_7030_df_ytrain = pd.DataFrame(binary_7030_y_train, columns = ['Label'])
binary_8020_df_ytrain = pd.DataFrame(binary_8020_y_train, columns = ['Label'])
binary_9010_df_ytrain = pd.DataFrame(binary_9010_y_train, columns = ['Label'])

multilabel_7030_df_ytrain = pd.DataFrame(multilabel_7030_y_train, columns = ['Label'])
multilabel_8020_df_ytrain = pd.DataFrame(multilabel_8020_y_train, columns = ['Label'])
multilabel_9010_df_ytrain = pd.DataFrame(multilabel_9010_y_train, columns = ['Label'])

def dummy_fun(doc):
    return doc

binary_7030_df_ytrain = pd.DataFrame(binary_7030_y_train, columns = ['Label'])
binary_7030_df_ytrain = binary_7030_df_ytrain.replace({'Label': mapping_binary})
binary_7030_train_y = binary_7030_df_ytrain['Label'].tolist()

binary_7030_tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)  

#classifier building/ fitting of training dataset to tfidf
binary_7030_fitted_training_x = binary_7030_tfidf.fit_transform(binary_7030_x_train)

#transform based on top 40 percent features
binary_7030_selector = SelectPercentile(f_classif, percentile = 40)
binary_7030_selector.fit(binary_7030_fitted_training_x, binary_7030_train_y)

binary_8020_df_ytrain = pd.DataFrame(binary_8020_y_train, columns = ['Label'])
binary_8020_df_ytrain = binary_8020_df_ytrain.replace({'Label': mapping_binary})
binary_8020_train_y = binary_8020_df_ytrain['Label'].tolist()

binary_8020_tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)  

#classifier building/ fitting of training dataset to tfidf
binary_8020_fitted_training_x = binary_8020_tfidf.fit_transform(binary_8020_x_train)

#transform based on top 40 percent features
binary_8020_selector = SelectPercentile(f_classif, percentile = 40)
binary_8020_selector.fit(binary_8020_fitted_training_x, binary_8020_train_y)

binary_9010_df_ytrain = pd.DataFrame(binary_9010_y_train, columns = ['Label'])
binary_9010_df_ytrain = binary_9010_df_ytrain.replace({'Label': mapping_binary})
binary_9010_train_y = binary_9010_df_ytrain['Label'].tolist()

binary_9010_tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)  

#classifier building/ fitting of training dataset to tfidf
binary_9010_fitted_training_x = binary_9010_tfidf.fit_transform(binary_9010_x_train)

#transform based on top 40 percent features
binary_9010_selector = SelectPercentile(f_classif, percentile = 40)
binary_9010_selector.fit(binary_9010_fitted_training_x, binary_9010_train_y)

multilabel_7030_df_ytrain = pd.DataFrame(multilabel_7030_y_train, columns = ['Label'])
multilabel_7030_df_ytrain = multilabel_7030_df_ytrain.replace({'Label': mapping_multilabel})
multilabel_7030_train_y = multilabel_7030_df_ytrain['Label'].tolist()

multilabel_7030_tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)  

#classifier building/ fitting of training dataset to tfidf
multilabel_7030_fitted_training_x = multilabel_7030_tfidf.fit_transform(multilabel_7030_x_train)

#transform based on top 40 percent features
multilabel_7030_selector = SelectPercentile(f_classif, percentile = 40)
multilabel_7030_selector.fit(multilabel_7030_fitted_training_x, multilabel_7030_train_y)

multilabel_8020_df_ytrain = pd.DataFrame(multilabel_8020_y_train, columns = ['Label'])
multilabel_8020_df_ytrain = multilabel_8020_df_ytrain.replace({'Label': mapping_multilabel})
multilabel_8020_train_y = multilabel_8020_df_ytrain['Label'].tolist()

multilabel_8020_tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)  

#classifier building/ fitting of training dataset to tfidf
multilabel_8020_fitted_training_x = multilabel_8020_tfidf.fit_transform(multilabel_8020_x_train)

#transform based on top 40 percent features
multilabel_8020_selector = SelectPercentile(f_classif, percentile = 40)
multilabel_8020_selector.fit(multilabel_8020_fitted_training_x, multilabel_8020_train_y)

multilabel_9010_df_ytrain = pd.DataFrame(multilabel_9010_y_train, columns = ['Label'])
multilabel_9010_df_ytrain = multilabel_9010_df_ytrain.replace({'Label': mapping_multilabel})
multilabel_9010_train_y = multilabel_9010_df_ytrain['Label'].tolist()

multilabel_9010_tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)  

#classifier building/ fitting of training dataset to tfidf
multilabel_9010_fitted_training_x = multilabel_9010_tfidf.fit_transform(multilabel_9010_x_train)

#transform based on top 40 percent features
multilabel_9010_selector = SelectPercentile(f_classif, percentile = 40)
multilabel_9010_selector.fit(multilabel_9010_fitted_training_x, multilabel_9010_train_y)

#load fasttext cnn models

tf.compat.v1.disable_eager_execution()

binary_7030_fasttext_model = load_model("binary/7030/fasttextcnn.h5")
binary_8020_fasttext_model = load_model("binary/8020/fasttextcnn.h5")
binary_9010_fasttext_model = load_model("binary/9010/fasttextcnn.h5")
multilabel_7030_fasttext_model = load_model("multilabel/7030/fasttextcnn.h5")
multilabel_8020_fasttext_model = load_model("multilabel/8020/fasttextcnn.h5")
multilabel_9010_fasttext_model = load_model("multilabel/9010/fasttextcnn.h5")

#load tfidf ffnn models
binary_7030_tfidf_model = load_model("binary/7030/cabasag_model.h5")
binary_8020_tfidf_model = load_model("binary/8020/cabasag_model.h5")
binary_9010_tfidf_model = load_model("binary/9010/cabasag_model.h5")
multilabel_7030_tfidf_model = load_model("multilabel/7030/cabasag_model.h5")
multilabel_8020_tfidf_model = load_model("multilabel/8020/cabasag_model.h5")
multilabel_9010_tfidf_model = load_model("multilabel/9010/cabasag_model.h5")


# In[6]:


#prediction functions

def fasttextcnn_predict(preprocessed_tweet, tokenizer, max_len, model, classification_type):
    output = ''
    tweet = " ".join(preprocessed_tweet)
    transformed = tokenizer.texts_to_sequences([tweet])
    padded = keras.preprocessing.sequence.pad_sequences(transformed, maxlen=max_len)

    if(classification_type == 0): 
        prediction = model.predict(padded)
        array = prediction.ravel()
        
        '\033[1m' + 'Python' + '\033[0m'
        if(array[0] > 0.5):
            output ="[{classification}: {value}]".format(classification = 'Hate', value = '{:.4f}'.format(array[0]))
        else:
            output = "[{classification}: {value}]".format(classification = 'Non-hate', value = '{:.4f}'.format(array[0]))
    else:
        prediction = model.predict(padded)
        array = prediction.ravel()
        highest = np.argmax(array)
        
        if highest == 0:
            output = "[{classification}: {value}], {classification2}: {value2}, {classification3}: {value3}".format(classification = 'Positive', value = '{:.4f}'.format(array[0]),
                                                                                                         classification2 = 'Negative', value2 ='{:.4f}'.format(array[1]),
                                                                                                         classification3 = 'Neutral', value3 = '{:.4f}'.format(array[2]))
        elif highest == 1:
            output = "{classification}: {value}, [{classification2}: {value2}], {classification3}: {value3}".format(classification = 'Positive', value = '{:.4f}'.format(array[0]),
                                                                                                         classification2 = 'Negative', value2 ='{:.4f}'.format(array[1]),
                                                                                                         classification3 = 'Neutral', value3 = '{:.4f}'.format(array[2]))
        elif highest == 2:
            output = "{classification}: {value}, {classification2}: {value2}, [{classification3}: {value3}]".format(classification = 'Positive', value = '{:.4f}'.format(array[0]),
                                                                                                         classification2 = 'Negative', value2 ='{:.4f}'.format(array[1]),
                                                                                                         classification3 = 'Neutral', value3 = '{:.4f}'.format(array[2]))
                
    return output

def tfidfffnn_predict(preprocessed_tweet, tfidf, selector, model, classification_type):
    output = ''
    
    transformed = tfidf.transform([preprocessed_tweet])
    vectorized_tweet = selector.transform(transformed).toarray()
    
    if(classification_type == 0): 
        prediction = model.predict(vectorized_tweet)
        array = prediction.ravel()
        highest = np.argmax(array)
        
        if highest == 0:
            output = "[{classification}: {value}], {classification2}: {value2}".format(classification = 'Non-hate', value = '{:.4f}'.format(array[0]), 
                                                                             classification2 = 'Hate', value2 = '{:.4f}'.format(array[1]))
        elif highest == 1:
            output = "{classification}: {value}, [{classification2}: {value2}]".format(classification = 'Non-hate', value = '{:.4f}'.format(array[0]), 
                                                                         classification2 = 'Hate', value2 = '{:.4f}'.format(array[1]))         
    else:
        prediction = model.predict(vectorized_tweet)
        array = prediction.ravel()
        highest = np.argmax(array)
        
        if highest == 0:
            output = "[{classification}: {value}], {classification2}: {value2}, {classification3}: {value3}".format(classification = 'Positive', value = '{:.4f}'.format(array[0]),
                                                                                                         classification2 = 'Negative', value2 ='{:.4f}'.format(array[1]),
                                                                                                         classification3 = 'Neutral', value3 = '{:.4f}'.format(array[2]))
        elif highest == 1:
            output = "{classification}: {value}, [{classification2}: {value2}], {classification3}: {value3}".format(classification = 'Positive', value = '{:.4f}'.format(array[0]),
                                                                                                         classification2 = 'Negative', value2 ='{:.4f}'.format(array[1]),
                                                                                                         classification3 = 'Neutral', value3 = '{:.4f}'.format(array[2]))
        elif highest == 2:
            output = "{classification}: {value}, {classification2}: {value2}, [{classification3}: {value3}]".format(classification = 'Positive', value = '{:.4f}'.format(array[0]),
                                                                                                         classification2 = 'Negative', value2 ='{:.4f}'.format(array[1]),
                                                                                                         classification3 = 'Neutral', value3 = '{:.4f}'.format(array[2]))
    return output


# In[ ]:





# In[ ]:


from tkinter import*
from tkinter import ttk  # Normal Tkinter.* widgets are not themed!
from ttkthemes import ThemedTk
    
thesis_gui = ThemedTk(theme="adapta")

thesis_gui.geometry("1200x900")
thesis_gui.configure(background = "GhostWhite")
thesis_gui.title("Philippine Election Related Tweets Sentiment Analysis")
thesis_gui.resizable(False, True)

style = ttk.Style()
style.configure("Bold.TLabel", font=("Helvetica", 10, "bold"))
style.configure("Bold2.TLabel", font=("Helvetica", 9, "bold"))
style.configure("Bold3.TLabel", font=("Helvetica", 8, "bold"))
style.configure("Bold.TButton", font=("Helvetica", 10, "bold"))

title = Label(thesis_gui, font = ('Helvetica 14 bold'), 
              bg = "GhostWhite",
              text = "Hate Speech in Filipino Election-Related Tweets: A Sentiment Analysis Using Convolutional Neural Networks")
title.pack(pady = 7)

names = Label(thesis_gui, font = ('Helvetica 12 italic'), 
              bg = "GhostWhite",
              text = "Argañosa, Marasigan, Villanueva, & Wenceslao. 2022")
names.pack()
#Input Label Frame
frame_label = ttk.Label(text="INPUT TWEET HERE", style="Bold.TLabel")
input_frame = ttk.LabelFrame(thesis_gui, 
                         labelwidget = frame_label,
                         relief='ridge')
input_frame.pack(fill="both", padx = 5, pady= 5)

inputtxt = Text(input_frame, height = 4,
                width = 200,
                font = ("Helvetica 11"),
                bg = "white")
inputtxt.pack( padx = 6, pady= 7)

#Preprocessing Label Frame
frame_label = ttk.Label(text="PREPROCESSING", style="Bold.TLabel")
preprocessing_frame = ttk.LabelFrame(thesis_gui, 
                         labelwidget = frame_label,
                         relief='ridge')
preprocessing_frame.pack(fill="both",padx = 5, pady= 5)

step_1 = StringVar()
step_1.set('')
#label
extraction_label = Label(preprocessing_frame, width=27,
                         text = "Extraction", 
                         font = ("Helvetica 10 bold"),
                         bg = 'light blue',relief = 'ridge')
extraction_label.grid(row=0,column=0,padx = 5, pady= 7)
#entry
extraction_entry = Entry(preprocessing_frame, width = 104, 
                         font = ("Helvetica 12 "),
                         bg = 'white',
                         textvariable = step_1, state = 'readonly', readonlybackground = 'white')
extraction_entry.grid(row=0,column=1,padx = 5, pady= 7, columnspan=2)

step_2 = StringVar()
step_2.set('')
#label
deidentification_label = Label(preprocessing_frame, width=27,
                         text = "Data De-Identification", 
                         font = ("Helvetica 10 bold"),
                         bg = 'light blue',relief = 'ridge')
deidentification_label.grid(row=1,column=0,padx = 5, pady= 7)
#entry
deidentification_entry = Entry(preprocessing_frame, width = 104, 
                         font = ("Helvetica 12"),
                         bg = 'white',
                         textvariable = step_2, state = 'readonly', readonlybackground = 'white')
deidentification_entry.grid(row=1,column=1,padx = 5, pady= 7, columnspan=2)

step_3 = StringVar()
step_3.set('')
#label
url_label = Label(preprocessing_frame, width=27,
                         text = "URL Removal", 
                         font = ("Helvetica 10 bold"),
                         bg = 'light blue',relief = 'ridge')
url_label.grid(row=2,column=0,padx = 5, pady= 7)
#entry
url_entry = Entry(preprocessing_frame, width = 104, 
                         font = ("Helvetica 12"),
                         bg = 'white',
                         textvariable = step_3, state = 'readonly', readonlybackground = 'white')
url_entry.grid(row=2,column=1,padx = 5, pady= 7, columnspan=2)


step_4 = StringVar()
step_4.set('')
#label
specialcharacter_label = Label(preprocessing_frame, width=27,
                         text = "Special Character Processing", 
                         font = ("Helvetica 10 bold"),
                         bg = 'light blue',relief = 'ridge')
specialcharacter_label.grid(row=3,column=0,padx = 5, pady= 7)
#entry
specialcharacter_entry = Entry(preprocessing_frame, width = 104, 
                         font = ("Helvetica 12"),
                         bg = 'white',
                         textvariable = step_4, state = 'readonly', readonlybackground = 'white')
specialcharacter_entry.grid(row=3,column=1,padx = 5, pady= 7, columnspan=2)


step_5 = StringVar()
step_5.set('')
#label
normalization_label = Label(preprocessing_frame, width=27,
                         text = "Normalization", 
                         font = ("Helvetica 10 bold"),
                         bg = 'light blue',relief = 'ridge')
normalization_label.grid(row=4,column=0,padx = 5, pady= 7)
#entry
normalization_entry = Entry(preprocessing_frame, width = 104, 
                         font = ("Helvetica 12"),
                         bg = 'white',
                         textvariable = step_5, state = 'readonly', readonlybackground = 'white')
normalization_entry.grid(row=4,column=1,padx = 5, pady= 7, columnspan=2)

step_6 = StringVar()
step_6.set('')
#label
hash_label = Label(preprocessing_frame, width=27,
                         text = "Hashtag Processing", 
                         font = ("Helvetica 10 bold"),
                         bg = 'light blue',relief = 'ridge')
hash_label.grid(row=5,column=0,padx = 5, pady= 7)
#entry
hash_entry = Entry(preprocessing_frame, width = 104, 
                         font = ("Helvetica 12"),
                         bg = 'white',
                         textvariable = step_6, state = 'readonly', readonlybackground = 'white')
hash_entry.grid(row=5,column=1,padx = 5, pady= 7, columnspan=2)

step_7 = StringVar()
step_7.set('')
#label
tokenization_label = Label(preprocessing_frame, width=27,
                         text = "Tokenization", 
                         font = ("Helvetica 10 bold"),
                         bg = 'light blue',relief = 'ridge')
tokenization_label.grid(row=6,column=0,padx = 5, pady= 7)
#entry
tokenization_entry = Entry(preprocessing_frame, width = 104, 
                         font = ("Helvetica 12"),
                         bg = 'white',
                         textvariable = step_7, state = 'readonly', readonlybackground = 'white')
tokenization_entry.grid(row=6,column=1,padx = 5, pady= 7, columnspan=2)


#classification frame
frame_label = ttk.Label(text="CLASSIFICATION", style="Bold.TLabel")
classification_frame = ttk.LabelFrame(thesis_gui, 
                         labelwidget = frame_label,
                         relief='ridge')
classification_frame.pack(fill="both",expand = 'yes', padx = 5, pady= 5)

#binary classification
frame_label = ttk.Label(text="Binary Classification", style="Bold2.TLabel")
binary_frame = ttk.LabelFrame(classification_frame, 
                         labelwidget = frame_label,
                         relief='sunken')
binary_frame.pack(side = 'left',fill="both",expand = 'yes', padx = 5, pady= 5)

#binary 7030

frame_label = ttk.Label(text="70:30 Split", style="Bold3.TLabel")
frame_7030 = ttk.LabelFrame(binary_frame, 
                         labelwidget = frame_label,
                         relief='sunken')
frame_7030.pack(fill="both",expand = 'yes', padx = 5, pady= 5)

#bf for binary fastText, bt for binary TFIDF
bf_7030 = StringVar()
bf_7030.set('')
bt_7030 = StringVar()
bt_7030.set('')
#label

bt_7030_label = Label(frame_7030, width=15,
                         text = "TFIDF FFNN", 
                         font = ("Helvetica 10"),relief = 'ridge',
                         bg = 'light yellow')
bt_7030_label.grid(row=0,column=0,padx = 5, pady= 7)
#classification
bt_7030_prediction = Label(frame_7030, width = 51, 
                         font = ("Helvetica 10"),
                         bg = 'white', relief = 'ridge',
                         textvariable = bt_7030)
bt_7030_prediction.grid(row=0,column=1,padx = 5, pady= 7)

#label
bf_7030_label = Label(frame_7030, width=15,
                         text = "fastText CNN", 
                         font = ("Helvetica 10"),relief = 'ridge',
                         bg = 'light pink')
bf_7030_label.grid(row=1,column=0,padx = 5, pady= 7)
#classification
bf_7030_prediction = Label(frame_7030, width = 51, 
                         font = ("Helvetica 10"),
                         bg = 'white', relief = 'ridge',
                         textvariable = bf_7030)
bf_7030_prediction.grid(row=1,column=1,padx = 5, pady= 7)

#binary 8020
bf_8020 = StringVar()
bf_8020.set('')
bt_8020 = StringVar()
bt_8020.set('')

frame_label = ttk.Label(text="80:20 Split", style="Bold3.TLabel")
frame_8020 = ttk.LabelFrame(binary_frame, 
                         labelwidget = frame_label,
                         relief='sunken')
frame_8020.pack(fill="both",expand = 'yes', padx = 5, pady= 5)

bt_8020_label = Label(frame_8020, width=15,
                         text = "TFIDF FFNN", 
                         font = ("Helvetica 10"),relief = 'ridge',
                         bg = 'light yellow')
bt_8020_label.grid(row=0,column=0,padx = 5, pady= 7)
#classification
bt_8020_prediction = Label(frame_8020, width = 51, 
                         font = ("Helvetica 10"),
                         bg = 'white', relief = 'ridge',
                         textvariable = bt_8020)
bt_8020_prediction.grid(row=0,column=1,padx = 5, pady= 7)

#label
bf_8020_label = Label(frame_8020, width=15,
                         text = "fastText CNN", 
                         font = ("Helvetica 10"),relief = 'ridge',
                         bg = 'light pink')
bf_8020_label.grid(row=1,column=0,padx = 5, pady= 7)
#classification
bf_8020_prediction = Label(frame_8020, width = 51, 
                         font = ("Helvetica 10"),
                         bg = 'white', relief = 'ridge',
                         textvariable = bf_8020)
bf_8020_prediction.grid(row=1,column=1,padx = 5, pady= 7)

#binary 9010
bf_9010 = StringVar()
bf_9010.set('')
bt_9010 = StringVar()
bt_9010.set('')

frame_label = ttk.Label(text="90:10 Split", style="Bold3.TLabel")
frame_9010 = ttk.LabelFrame(binary_frame, 
                         labelwidget = frame_label,
                         relief='sunken')
frame_9010.pack(fill="both",expand = 'yes', padx = 5, pady= 5)

bt_9010_label = Label(frame_9010, width=15,
                         text = "TFIDF FFNN", 
                         font = ("Helvetica 10"),relief = 'ridge',
                         bg = 'light yellow')
bt_9010_label.grid(row=0,column=0,padx = 5, pady= 7)
#classification
bt_9010_prediction = Label(frame_9010, width = 51, 
                         font = ("Helvetica 10"),
                         bg = 'white', relief = 'ridge',
                         textvariable = bt_9010)
bt_9010_prediction.grid(row=0,column=1,padx = 5, pady= 7)

#label
bf_9010_label = Label(frame_9010, width=15,
                         text = "fastText CNN", 
                         font = ("Helvetica 10"),relief = 'ridge',
                         bg = 'light pink')
bf_9010_label.grid(row=1,column=0,padx = 5, pady= 7)
#classification
bf_9010_prediction = Label(frame_9010, width = 51, 
                         font = ("Helvetica 10"),
                         bg = 'white', relief = 'ridge',
                         textvariable = bf_9010)
bf_9010_prediction.grid(row=1,column=1,padx = 5, pady= 7)

 
#multilabel classification
frame_label = ttk.Label(text="Multi label Classification", style="Bold2.TLabel")
multilabel_frame = ttk.LabelFrame(classification_frame, 
                         labelwidget = frame_label,
                         relief='sunken')
multilabel_frame.pack(side = 'right',fill="both",expand = 'yes', padx = 5, pady= 5)

#variables
mf_7030m = StringVar()
mf_7030m.set('')
mt_7030m = StringVar()
mt_7030m.set('')
mf_8020m = StringVar()
mf_8020m.set('')
mt_8020m = StringVar()
mt_8020m.set('')
mf_9010m = StringVar()
mf_9010m.set('')
mt_9010m = StringVar()
mt_9010m.set('')

#multilabel 7030
frame_label = ttk.Label(text="70:30 Split", style="Bold3.TLabel")
frame_7030m = ttk.LabelFrame(multilabel_frame, 
                         labelwidget = frame_label,
                         relief='sunken')
frame_7030m.pack(fill="both",expand = 'yes', padx = 5, pady= 5)

#label
mt_7030_label = Label(frame_7030m, width=15,
                         text = "TFIDF FFNN", 
                         font = ("Helvetica 10"), relief = 'ridge',
                         bg = 'light yellow')
mt_7030_label.grid(row=0,column=0,padx = 5, pady= 7)
#entry
mt_7030_prediction = Label(frame_7030m, width = 51, 
                         font = ("Helvetica 10"), relief = 'ridge',
                         bg = 'white',
                         textvariable = mt_7030m)
mt_7030_prediction.grid(row=0,column=1,padx = 5, pady= 7)

#label
mf_7030_label = Label(frame_7030m, width=15,
                         text = "fastText CNN", 
                         font = ("Helvetica 10"), relief = 'ridge',
                         bg = 'light pink')
mf_7030_label.grid(row=1,column=0,padx = 5, pady= 7)
#entry
mf_7030_prediction = Label(frame_7030m, width = 51, 
                         font = ("Helvetica 10"), relief = 'ridge',
                         bg = 'white',
                         textvariable = mf_7030m)
mf_7030_prediction.grid(row=1,column=1,padx = 5, pady= 7)

#multilabel 8020
frame_label = ttk.Label(text="80:20 Split", style="Bold3.TLabel")
frame_8020m = ttk.LabelFrame(multilabel_frame, 
                         labelwidget = frame_label,
                         relief='sunken')
frame_8020m.pack(fill="both",expand = 'yes', padx = 5, pady= 5)

#label
mt_8020_label = Label(frame_8020m, width=15,
                         text = "TFIDF FFNN", 
                         font = ("Helvetica 10"), relief = 'ridge',
                         bg = 'light yellow')
mt_8020_label.grid(row=0,column=0,padx = 5, pady= 7)
#entry
mt_8020_prediction = Label(frame_8020m, width = 51, 
                         font = ("Helvetica 10"), relief = 'ridge',
                         bg = 'white',
                         textvariable = mt_8020m)
mt_8020_prediction.grid(row=0,column=1,padx = 5, pady= 7)

#label
mf_8020_label = Label(frame_8020m, width=15,
                         text = "fastText CNN", 
                         font = ("Helvetica 10"), relief = 'ridge',
                         bg = 'light pink')
mf_8020_label.grid(row=1,column=0,padx = 5, pady= 7)
#entry
mf_8020_prediction = Label(frame_8020m, width = 51, 
                         font = ("Helvetica 10"), relief = 'ridge',
                         bg = 'white',
                         textvariable = mf_8020m)
mf_8020_prediction.grid(row=1,column=1,padx = 5, pady= 7)

#multilabel 9010
frame_label = ttk.Label(text="90:10 Split", style="Bold3.TLabel")
frame_9010m = ttk.LabelFrame(multilabel_frame, 
                         labelwidget = frame_label,
                         relief='sunken')
frame_9010m.pack(fill="both",expand = 'yes', padx = 5, pady= 5)

#label
mt_9010_label = Label(frame_9010m, width=15,
                         text = "TFIDF FFNN", 
                         font = ("Helvetica 10"), relief = 'ridge',
                         bg = 'light yellow')
mt_9010_label.grid(row=0,column=0,padx = 5, pady= 7)
#entry
mt_9010_prediction = Label(frame_9010m, width = 51, 
                         font = ("Helvetica 10"), relief = 'ridge',
                         bg = 'white',
                         textvariable = mt_9010m)
mt_9010_prediction.grid(row=0,column=1,padx = 5, pady= 7)

#label
mf_9010_label = Label(frame_9010m, width=15,
                         text = "fastText CNN", 
                         font = ("Helvetica 10"), relief = 'ridge',
                         bg = 'light pink')
mf_9010_label.grid(row=1,column=0,padx = 5, pady= 7)
#entry

mf_9010_prediction = Label(frame_9010m, width = 51, 
                         font = ("Helvetica 10 "), relief = 'ridge',
                         bg = 'white',
                         textvariable = mf_9010m)
mf_9010_prediction.grid(row=1,column=1,padx = 5, pady= 7)

#buttons
clear_button = ttk.Button(thesis_gui, text="Clear", 
                             width="20",command = clear, style = "Bold.TButton")
clear_button.pack(side='left',padx=5,pady=3, fill='both', expand = 'yes')

classify_button = ttk.Button(thesis_gui, text="Classify Tweet", 
                             width="20",command = classify, style = "Bold.TButton")
classify_button.pack(side='right',padx=5,pady=3, fill='both', expand = 'yes')

#run gui
thesis_gui.mainloop()


# In[ ]:





# In[ ]:




