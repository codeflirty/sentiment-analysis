# ============================================================================
# ------------------------ MODEL TRAINING AND TESTING ------------------------
# ============================================================================

# Step 1:
# ================
# Imoprt Libraries
# ================

import os
import nltk
import joblib
nltk.download()
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
nltk.download('punkt')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')
from prettytable import PrettyTable
from nltk.tokenize import RegexpTokenizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2:
# ============
# Load Dataset
# ============

dataset = pd.read_csv("Dataset/data.csv")

# Step 3:
# =================
# Visualize Dataset
# =================

Sentiment_val=dataset.groupby('Feeling').count()
plt.bar(Sentiment_val.index.values, Sentiment_val['Tweets'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.savefig('Graphs/Data_2_Count.png')
plt.show()

# Step 4:
# ============
# Tokenization
# ============

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(dataset['Tweets'])
tf=TfidfVectorizer()
text_tf= tf.fit_transform(dataset['Tweets'])

# Step 5:
# ================
# Train Test Split
# ================

x=text_tf
y=dataset['Feeling']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)

# Step 6:
# ===================
# Build Model ( MLP )
# ===================

model = MLPClassifier(verbose=1)

# Step 7:
# ===========
# Train Model
# ===========

model.fit(x_train,y_train)

# Step 8:
# =============================
# Evaluate Performance of Model
# =============================

preds = model.predict(x_test)
test_labels = y_test

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
plt.savefig('Graphs/Model_2_Confusion_Matrix.png')
plt.show()

# Step 8:
# ==================
# Save Trained Model
# ==================

joblib.dump(model, 'Supporting Material/trained_model_2.pkl')

print("\n\n=======================")
print("Trained Model is Saved!")
print("=======================\n\n")

# ============================================================================
# ---------------------------- THANK YOU SO MUCH -----------------------------
# ============================================================================