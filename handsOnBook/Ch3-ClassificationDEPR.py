#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 21:56:34 2026

@author: singh
print("SOURCE:https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html")
"""

import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

print("\n-----Digits dataset-----\n")

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
    
print("\n-----Classification-----\n")
#flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples), -1)

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5,shuffle=False
)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

print("\n visualise the first 4 test samples\n")
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
plt.show()

print("\n Print the main classification metrics\n")
print(
      f"Classification report for classifier {clf}: \n"
      f"{metrics.classification_report(y_test, predicted)}\n"
      )

print("\n Plot confusion matrix\n")
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

print("\n Print the main classification metrics - not in terms of y_true and y_pred\n")
# The ground truth and predicted lists
y_true = []
y_pred = []
cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]
        
print(
     "Classification report rebuilt from confusion matrix:\n"
     f"{metrics.classification_report(y_true, y_pred)}\n"
)

print("\n-----Precision/Recall Trade-off with SGDClassfier and decision function-----\n")
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict

sgd_clf = SGDClassifier()
y_scores= cross_val_predict(sgd_clf,X_train,y_train, cv=3, method='decision_function')
print("\ny_scores:\n",y_scores)
threshold =0
y_some_digit_pred = (y_scores > threshold)
print("(y_scores \> threshold=0 ):\n",y_some_digit_pred)
threshold =8000
y_some_digit_pred = (y_scores > threshold)
print("(y_scores \> threshold=8000 ):\n",y_some_digit_pred)
print("\nDeciding which threshold to use: using decision_function")
y_scores=cross_val_predict(sgd_clf, X_train, y_train, cv=3, method="decision_function")
print("\ncross_val_predict(sgd_clf, X_train, y_train, cv=3, method='decision_function')\n",y_scores)

print("\nUse precision_recall_curve() function for all thresholds\n")
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

def plot_precision_recall_vs_threhold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:1],"b--", label="Precision")
    plt.plot(thresholds, recalls[:1],"g--", label="Recall")
    

plot_precision_recall_vs_threhold(precisions, recalls, thresholds)
plt.show()