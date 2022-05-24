# ============================================================================
# ------------------------ MODEL TRAINING AND TESTING ------------------------
# ============================================================================

# Step 1:
# ================
# Imoprt Libraries
# ================

import nlp
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Step 3:
# ===============
# Tokenize Tweets
# ===============

tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')

tokenizer.fit_on_texts(tweets)

# Step 4:
# ==================
# Truncate Sequences
# ==================

lengths = [len(t.split(' ')) for t in tweets]

plt.hist(lengths, bins=len(set(lengths)))
plt.title("Lengths")
plt.show()

def get_sequences(tokenizer, tweets):
    sequences = tokenizer.texts_to_sequences(tweets)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=50, padding='post')
    return padded_sequences

padded_train_sequences = get_sequences(tokenizer, tweets)

# Step 5:
# =====================
# Prepare Output Labels
# =====================

classes = set(labels)

countplt, ax = plt.subplots(figsize = (10,5))
ax = sns.countplot(labels, palette='Set3')
plt.title("Labels")
plt.savefig('Graphs/Data_1_Count.png')
plt.show()

A = pd.DataFrame(labels)
A['sentiment'] = labels

col = 'sentiment'
fig, (ax1, ax2)  = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
explode = list((np.array(list(A[col].dropna().value_counts()))/sum(list(A[col].dropna().value_counts())))[::-1])[:]
lab = A[col].value_counts()[:].index
sizes = A[col].value_counts()[:]
ax2.pie(sizes,  explode=explode, startangle=60, labels=lab, autopct='%1.0f%%', pctdistance=0.8)
ax2.add_artist(plt.Circle((0,0),0.4,fc='white'))
sns.countplot(y =col, data = A, ax=ax1)
ax1.set_title("Count of Each Emotion")
ax2.set_title("Percentage of Each Emotion")
plt.savefig('Graphs/Percentage of Each Emotion.png')
plt.show()

classes_to_index = dict((c, i) for i, c in enumerate(classes))
index_to_classes = dict((v, k) for k, v in classes_to_index.items())

names_to_ids = lambda labels: np.array([classes_to_index.get(x) for x in labels])

train_labels = names_to_ids(labels)

# Step 6:
# ===================
# Build Model ( RNN )
# ===================

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=50),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Step 7:
# ===========
# Train Model
# ===========

val_tweets, val_labels = get_tweets(val)
val_sequences = get_sequences(tokenizer, val_tweets)
val_labels = names_to_ids(val_labels)

history = model.fit( padded_train_sequences, train_labels, validation_data=(val_sequences, val_labels), epochs=25,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)])

# Step 8:
# =============================
# Evaluate Performance of Model
# =============================

def show_history(h):
    epochs_trained = len(history.history['loss'])
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs_trained), history.history.get('accuracy'), label='Training')
    plt.plot(range(0, epochs_trained), history.history.get('val_accuracy'), label='Validation')
    plt.ylim([0., 1.])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs_trained), history.history.get('loss'), label='Training')
    plt.plot(range(0, epochs_trained), history.history.get('val_loss'), label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Graphs/Model_1_Performance.png')
    plt.show()


show_history(history)

test_tweets, test_labels = get_tweets(test)
test_sequences = get_sequences(tokenizer, test_tweets)
test_labels = names_to_ids(test_labels)

eval = model.evaluate(test_sequences, test_labels)

for a in range(0,10):
    i = random.randint(0, len(test_labels) - 1)
    print('\n\n======')
    print('Tweet:')
    print('======')
    print(test_tweets[i])
    print("\n")
    
    p = np.argmax(model.predict(np.expand_dims(test_sequences[i], axis=0)), axis=-1)[0]
    pp = model.predict(np.expand_dims(test_sequences[i], axis=0), batch_size=1)
    
    x = PrettyTable()
    x.field_names = ["Title", "Value"]
    x.add_row(["Actual Emotion", index_to_classes[test_labels[i]]])
    x.add_row(["Predicted Emotion", index_to_classes.get(p)])
    x.add_row(["Confidence", str(round(np.max(pp*100),2))+'%'])
    print(x)

preds = np.argmax(model.predict(test_sequences), axis=-1)
preds.shape, test_labels.shape

model_accuracy = accuracy_score(test_labels, preds)
print("\n\n===============")
print("Accuracy Score:")
print("===============\n\n")
print("  Accuracy: ", round(model_accuracy*100,2), "%")

model_report = classification_report(test_labels, preds)
print("\n\n======================")
print("Classification Report:")
print("======================\n\n")
print(model_report)

model_confusion_matrix = confusion_matrix(test_labels, preds)
print("\n\n=================")
print("Confusion Matrix:")
print("=================\n\n")
print(model_confusion_matrix)

print("\n\n===============================")
print("Confusion Matrix with Heat MAP:")
print("===============================\n\n")
model_confusion_matrix_heatmap = confusion_matrix(test_labels, preds, normalize = 'true')
sns.set(rc={'figure.figsize':(10, 6)})
sns.heatmap(model_confusion_matrix_heatmap, annot=True)
plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], labels=classes)
plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], labels=classes)
plt.savefig('Graphs/Model_1_Confusion_Matrix.png')
plt.show()

# Step 8:
# ==================
# Save Trained Model
# ==================

model.save('Supporting Material/trained_model_1.h5')

print("\n\n=======================")
print("Trained Model is Saved!")
print("=======================\n\n")

# ============================================================================
# ---------------------------- THANK YOU SO MUCH -----------------------------
# ============================================================================