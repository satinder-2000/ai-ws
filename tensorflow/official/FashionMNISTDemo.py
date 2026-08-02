#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 06:21:07 2026

@author: singh
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

print("-----Import the Fashion MNIST dataset-----")
print()
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels)=fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("-----Ixplore the data-----")
print()

print("train_images.shape:\n", train_images.shape)
print()
print("train_labels:\n", train_labels)
print()
print("test_images.shape:\n", test_images.shape)
print()
print("len(test_labels:\n", len(test_labels))
print()

print("-----Preprocess the data-----")
print()
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

print("-----Scale these values to a range of 0 to 1 before feeding them to the neural network model.-----")
print()
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()

print("-----Build the model - set up the layers.-----")
print()
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

print("-----Compile the model -----")
print()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("-----Train the model -----")
print()
model.fit(train_images, train_labels, epochs=10)


print("-----Evaluate accuracy -----")
print()
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)

print("-----Make predictions -----")
print()

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)
print("predictions[0] :\n",predictions[0])
print("np.argmax(predictions[0]):\n", np.argmax(predictions[0]))
print("test_labels[0]:\n", test_labels[0])

print("-----graph the full set of 10 class predictions.-----")
print()

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmp=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
        
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color=color)
    
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    
print("-----Verify predictions -----")
print()
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()