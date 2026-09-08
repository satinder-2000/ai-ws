#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:32:19 2026

@author: singh
"""
print("Source:https://medium.com/@piyushkashyap045/understanding-precision-recall-and-f1-score-metrics-ea219b908093")
print("\n-----Import Libraries-----\n")
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

print("\n-----Train a Classifier-----\n")
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#load database
data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(data.data,
                    data.target, test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print("\n-----Compute Metrics-----\n")
print("Precision (Per Class):", precision_score(y_test, y_pred, average=None))
print("Recall (Per Class):", recall_score(y_test, y_pred, average=None))
print("F1 Score (Per Class):", f1_score(y_test, y_pred, average=None))

print("\nMacro and Weighted Averages\n")
print("Macro Precision):", precision_score(y_test, y_pred, average='macro'))
print("Weighted Precision):", precision_score(y_test, y_pred, average='weighted'))
print("Macro Recall:", recall_score(y_test, y_pred, average='macro'))
print("Weighted Recall:", recall_score(y_test, y_pred, average='weighted'))
print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Weighted F1 Score:", f1_score(y_test, y_pred, average='weighted'))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))