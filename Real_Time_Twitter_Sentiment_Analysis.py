# ========================================================================
# ----------------- REAL TIME TWITTER SENTIMENT ANALYSIS -----------------
# ========================================================================

# Step 1:
# ================
# Imoprt Libraries
# ================

import os
import re
import sys
import nlp
import time
import nltk
import joblib
import string
import warnings
import numpy as np
import pandas as pd
from tweepy import API
import tensorflow as tf
from tweepy import Stream
from datetime import datetime
from tweepy import OAuthHandler
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from googletrans import Translator
from nltk.tokenize import RegexpTokenizer
from tweepy.streaming import StreamListener
from urllib3.exceptions import ProtocolError
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 2:
# ============
# Load Dataset
# ============

dataset = nlp.load_dataset('emotion')

train = dataset['train']
val = dataset['validation']
test = dataset['test']

def get_tweets(data):
    tweets = [x['text'] for x in data]
    labels = [x['label'] for x in data]
    return tweets, labels

tweets, labels = get_tweets(train)

dataset_2 = pd.read_csv("Dataset/data.csv")

# Step 3:
# ===============
# Tokenize Tweets
# ===============

tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')

tokenizer.fit_on_texts(tweets)

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(dataset_2['Tweets'])
tfi=TfidfVectorizer()
text_tf= tfi.fit_transform(dataset_2['Tweets'])

# Step 4:
# ==================
# Truncate Sequences
# ==================

lengths = [len(t.split(' ')) for t in tweets]

def get_sequences(tokenizer, tweets):
    sequences = tokenizer.texts_to_sequences(tweets)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=50, padding='post')
    return padded_sequences

padded_train_sequences = get_sequences(tokenizer, tweets)

padded_train_sequences[10]

# Step 5:
# =====================
# Prepare Output Labels
# =====================

classes = set(labels)

A = pd.DataFrame(labels)
A['sentiment'] = labels

classes_to_index = dict((c, i) for i, c in enumerate(classes))
index_to_classes = dict((v, k) for k, v in classes_to_index.items())

names_to_ids = lambda labels: np.array([classes_to_index.get(x) for x in labels])

train_labels = names_to_ids(labels)

# Step 6:
# ==================
# Load Trained Model
# ==================

model_1 = tf.keras.models.load_model('Supporting Material/trained_model_1.h5')
model_2 = joblib.load('Supporting Material/trained_model_2.pkl')

# Step 7:
# ====================================
# Translate other Languages to English
# ====================================

def translation(sentence):
    try:
        if(len(str(sentence))<1):
            return "Sorry, No Tweet Extracted"
        else:
            translator = Translator()
            text = translator.translate(str(sentence), dest="en").text
            return text
    except:
        return "Sorry: Error in Translation"

# Step 8:
# ============
# Clean Tweets
# ============

def clean_text(text):
    regex_html = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    remove_digits = str.maketrans('', '', string.digits + string.punctuation)
    text = re.sub(regex_html, '', text)
    text = text.translate(remove_digits)
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split()).lower()

# Step 9:
# ==============================
# Live Tweets Sentiment Analysis
# ==============================

def clear():
    os.system( 'cls' )
clear()

consumer_key = 'UOi2wC2onaoyaD0pc5Vteo4v3'
consumer_secret = 'SVclTIfwp6sheQydF8trsAxMpwefle9p0nOQpE9pNzJcU1eyFu'
access_token = '1444438055595622400-DeVCMTFAw7T2Sr8YoxIAZY4DD3mhJo'
access_token_secret = 'b5xkLuS1RqkrAWoTuEL8UORWmKp75wREsvgmrNrzKFsiw'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = API(auth)


class Listener(StreamListener):

    df = pd.DataFrame()
    def on_status(self, status):
        
        new_tweet = status.text
        if(len(new_tweet) >= 2):
            if (new_tweet[0] != "R" and new_tweet[1] != "T"):
                translated_tweet = translation(new_tweet)
                
                process_tweet = clean_text(translated_tweet)
                
                new_sequences = get_sequences(tokenizer, [process_tweet])

                test_p = np.argmax(model_1.predict(new_sequences))
                test_pp = model_1.predict(new_sequences)

                sentiment_value = index_to_classes.get(test_p)

                text_tf= tfi.transform([process_tweet])

                test_p_2 = np.argmax(model_2.predict_proba(text_tf))
                test_pp_2 = model_2.predict_proba(text_tf)

                a = '{:04.1f}'.format(test_pp[0][1]*100)
                b = '{:04.1f}'.format(test_pp[0][4]*100)
                c = '{:04.1f}'.format(test_pp[0][2]*100)
                d = '{:04.1f}'.format(test_pp[0][5]*100)
                e = '{:04.1f}'.format(test_pp[0][3]*100)
                f = '{:04.1f}'.format(test_pp[0][0]*100)

                a2 = '{:04.1f}'.format(test_pp_2[0][0]*100)
                b2 = '{:04.1f}'.format(test_pp_2[0][1]*100)
                c2 = '{:04.1f}'.format(test_pp_2[0][2]*100)
                d2 = '{:04.1f}'.format(test_pp_2[0][3]*100)
                e2 = '{:04.1f}'.format(test_pp_2[0][4]*100)
                f2 = '{:04.1f}'.format(test_pp_2[0][5]*100)

                avg_1 = '{:04.1f}'.format((test_pp[0][1]*100 + test_pp_2[0][0]*100)/2)
                avg_2 = '{:04.1f}'.format((test_pp[0][4]*100 + test_pp_2[0][1]*100)/2)
                avg_3 = '{:04.1f}'.format((test_pp[0][2]*100 + test_pp_2[0][2]*100)/2)
                avg_4 = '{:04.1f}'.format((test_pp[0][5]*100 + test_pp_2[0][3]*100)/2)
                avg_5 = '{:04.1f}'.format((test_pp[0][3]*100 + test_pp_2[0][4]*100)/2)
                avg_6 = '{:04.1f}'.format((test_pp[0][0]*100 + test_pp_2[0][5]*100)/2)
                
                output = open("Supporting Material/Output.csv", "a")
                output.write(str(datetime.now().strftime("%H:%M:%S"))+","+str(avg_1)+","+str(avg_2)+","+str(avg_3)+","+str(avg_4)+","+str(avg_5)+","+str(avg_6))
                output.write('\n')
                output.close()
                output = open("Supporting Material/Output.txt", "a")
                output.write(str(avg_1)+","+str(avg_2)+","+str(avg_3)+","+str(avg_4)+","+str(avg_5)+","+str(avg_6))
                output.write('\n')
                output.close()
                
                df = pd.read_csv("Supporting Material/Output.csv")

                pre = 0
                flag = 0
                list_0 = []
                list_1 = []
                list_2 = []
                list_3 = []
                list_4 = []
                list_5 = []
                list_6 = []
                
                for i in range(len(df)):
                    time_string = df.loc[len(df)-i-1][0]
                    date_time = datetime.strptime(time_string, "%H:%M:%S")
                    a_timedelta = date_time - datetime(1900, 1, 1)
                    seconds = a_timedelta.total_seconds()
                    if(flag == 0):
                        pre = seconds
                        flag = 1
                    diff = pre-seconds
                    if(diff<30):
                        list_0.append(df.loc[len(df)-i-1][0])
                        list_1.append(df.loc[len(df)-i-1][1])
                        list_2.append(df.loc[len(df)-i-1][2])
                        list_3.append(df.loc[len(df)-i-1][3])
                        list_4.append(df.loc[len(df)-i-1][4])
                        list_5.append(df.loc[len(df)-i-1][5])
                        list_6.append(df.loc[len(df)-i-1][6])
                    else:
                        break

                clear()
                sys.stdout.write("\r" "**********************************************************************\n") 
                sys.stdout.write("\r" "     Average(30 sec)         -->        " + list_0[-1] + "   to   " + list_0[0] + "\n")
                sys.stdout.write("\r" "**********************************************************************\n")
                sys.stdout.write("\r" " Anger:      Love:       Fear:       Happiness:  Sadness:    Surprise:   \n")
                sys.stdout.write("\r" " "+str('{:04.1f}'.format(np.mean(list_1)))+ "        "+str('{:04.1f}'.format(np.mean(list_2)))+ "        "+str('{:04.1f}'.format(np.mean(list_3)))+ "        "+str('{:04.1f}'.format(np.mean(list_4)))+ "        "+str('{:04.1f}'.format(np.mean(list_5)))+ "        "+str('{:04.1f}'.format(np.mean(list_6)))+ "\n" )
                
                aa7 = str(new_tweet.replace("\r","").replace("\n","").replace("\t","").replace(","," ")[:75])

                sys.stdout.write("\r" "\n\n***************\n") 
                sys.stdout.write("\r" "Extracted Tweet\n")
                sys.stdout.write("\r" "***************\n")
                sys.stdout.write("\r" "" + aa7 + "\n\n")
                sys.stdout.write("\r" " Emotion:           Model 1           Model 2           Average\n")
                sys.stdout.write("\r" " Anger:             " + str(a) + " %" + "            " + str(a2) + " %" + "            " + str(avg_1) + " %" +
                                    "\n Love:              " + str(b) + " %" + "            " + str(b2) + " %" + "            " + str(avg_2) + " %" + 
                                    "\n Fear:              " + str(c) + " %" + "            " + str(c2) + " %" + "            " + str(avg_3) + " %" +  
                                    "\n Happiness:         " + str(d) + " %" + "            " + str(d2) + " %" + "            " + str(avg_4) + " %" + 
                                    "\n Sadness:           " + str(e) + " %" + "            " + str(e2) + " %" + "            " + str(avg_5) + " %" +  
                                    "\n Surprise:          " + str(f) + " %" + "            " + str(f2) + " %" + "            " + str(avg_6) + " %" +
                                    "\n" )
                
                aa1 = str('{:04.1f}'.format(np.mean(list_1)))
                aa2 = str('{:04.1f}'.format(np.mean(list_2)))
                aa3 = str('{:04.1f}'.format(np.mean(list_3)))
                aa4 = str('{:04.1f}'.format(np.mean(list_4)))
                aa5 = str('{:04.1f}'.format(np.mean(list_5)))
                aa6 = str('{:04.1f}'.format(np.mean(list_6)))
                
                output = open("Supporting Material/Average.csv", "a", encoding="utf-8")
                output.write(str(aa1)+","+str(aa2)+","+str(aa3)+","+str(aa4)+","+str(aa5)+","+str(aa6)+","+str(aa7))
                output.write('\n')
                output.close()

                sys.stdout.flush()
    
    def on_error(self, status_code):
        print(status_code)
        return False

listener = Listener()
stream = Stream(auth=api.auth, listener=listener)

import requests
from urllib3.exceptions import ProtocolError

while True:
    try:
        stream.sample()
    except (ProtocolError, AttributeError):
        continue

# ========================================================================
# ---------------------------- THANK YOU SO MUCH -------------------------
# ========================================================================