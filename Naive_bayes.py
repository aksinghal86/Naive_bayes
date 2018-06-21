# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 19:46:28 2018

@author: asinghal
"""

import pandas as pd
import string
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_table('/Users/asinghal/Documents/Projects/Naive_Bayes/SMSSpamCollection', 
                   sep = '\t', names = ['label', 'sms_message'])

# Convert the y column into binary (required for prediction metrics)
df['label'] = df.label.map({'ham':0, 'spam':1})
df.head()

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state = 1)

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data (BoW) and return as a matrix
# Also converts to lowercase, removes punctuations and tokenizes
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return as a matrix 
testing_data = count_vector.transform(X_test)

# Instantiate MultinomialNB() for classifier problem
naive_bayes = MultinomialNB()

# Fit the training data
naive_bayes.fit(training_data, y_train)

# Make predictions on testing data
predictions = naive_bayes.predict(testing_data)

# Prediction metrics
print('Accuracy score: {:.5}'.format(accuracy_score(y_test, predictions)))
print('Precision score: {:.5}'.format(precision_score(y_test, predictions)))
print('Recall score: {:.5}'.format(recall_score(y_test, predictions)))
print('F1 score: {:.5}'.format(f1_score(y_test, predictions)))

