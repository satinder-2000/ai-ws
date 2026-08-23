#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 23:47:56 2026

@author: singh
"""
print("Multiple Instance Learning > MIL")
print()
print("-----Setup-----")
import numpy as np
import keras
from keras import layers
from keras import ops
from tqdm import tqdm
from matplotlib import pyplot as plt

plt.style.use("ggplot")

print()
print("-----Create dataset-----")
print()
print("---Configuration parameters---")
print()
POSITIVE_CLASS = 1
BAG_COUNT = 1000
VAL_BAG_COUNT = 300
BAG_SIZE = 3
PLOT_SIZE = 3
ENSEMBLE_AVG_COUNT = 1

print()
print("---Prepare bags---")
print()
def create_bags(input_data, input_labels, positive_class, bag_count, instance_count):
    #Set up bags
    bags = []
    bag_labels = []
    
    #Normalize input data
    input_data = np.divide(input_data, 255.0)
    
    # Count positive samples
    count = 0
    
    for _ in range(bag_count):
       # Pick a fixed size random subset of samples.
       index = np.random.choice(input_data.shape[0], instance_count, replace= False)
       instances_data=input_data[index]
       instances_labels = input_labels[index]
       
       # By default, all bags are labeled as 0.
       bag_label = 0
       
       # Check if there is at least a positive class in the bag.
       if positive_class in instances_labels:
           # Positive bag will be labeled as 1.
           bag_label = 1
           count += 1
           
       bags.append(instances_data)
       bag_labels.append(np.array([bag_label]))
       
     
    print(f"Positive bags: {count}")
    print(f"Negative bags: {bag_count - count}")
    
    return(list(np.swapaxes(bags, 0, 1)), np.array(bag_labels))

# load tha MNIST dataset
(x_train, y_train),(x_val, y_val) = keras.datasets.mnist.load_data()

#create training data
train_data, train_labels = create_bags(
    x_train, y_train, POSITIVE_CLASS, BAG_COUNT, BAG_SIZE
)
val_data, val_labels = create_bags(
    x_val, y_val, POSITIVE_CLASS, VAL_BAG_COUNT, BAG_SIZE
)

print()
print("-----Create the model-----")
print()
print("---Attention operator implementation---")

class MILAttentionLayer(layers.Layer):
    """Implementation of the attention-based Deep MIL layer.

    Args:
      weight_params_dim: Positive Integer. Dimension of the weight matrix.
      kernel_initializer: Initializer for the `kernel` matrix.
      kernel_regularizer: Regularizer function applied to the `kernel` matrix.
      use_gated: Boolean, whether or not to use the gated mechanism.

    Returns:
      List of 2D tensors with BAG_SIZE length.
      The tensors are the attention scores after softmax with shape `(batch_size, 1)`.
    """
    
    def __init__(
            self,
            weight_params_dim,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            use_gated=False,
            **kwargs
        ):
        super().__init__(**kwargs)
        
        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated
        
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer
        
        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer
        
    
    def build(self, input_shape):
        #Input shape
        #List of 2D tensors with shape: (batch_size, input_dim)
        input_dim = input_shape[0][1]
        
        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True
        )
        
        self.w_weight_params = self.add_weight(
            shape=(self.weight_params_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w.w_regularizer,
            trainable=True
        )
        
        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.weight_params_dim),
                initializer=self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True
           )
        else:
            self.u_weight_params = None
            
        self.input_built = True    
        
    
    def call(self, inputs):
        # Assigning variables from the number of inputs.
        instances = [self.compute_attantion_scores(instance) for instance in inputs]
        
        # Stack instances into a single tensor.
        instances = ops.stack(instances)
        
        # Apply softmax over instances such that the output summation is equal to 1.
        alpha = ops.softmax(instances, axis=0)
        
        # Split to recreate the same array of tensors we had as inputs.
        return [alpha[i] for i in range(alpha.shape[0])]
        
    
    def compute_attantion_scores(self, instance):
        # Reserve in-case "gated mechanism" used.
        original_instance = instance
        
        # tanh(v*h_k^T)
        instance = ops.tanh(ops.tensordot(instance, self.v_weight_params, axes=1))
        
        # for learning non-linear relations efficiently.
        if self.use_gated:
            instance = instance * ops.sigmoid(
                ops.tensordot(original_instance, self.u_weight_params, axes=1)
            )
            
        # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
        return ops.tensordot(instance, self.w_weight_params, axes=1)


print("-----Visulizer tool-----")
print()
def plot(data, labels, bag_class, predictions=None, attention_weights=None):
    """ "Utility for plotting bags and attention weights.

    Args:
      data: Input data that contains the bags of instances.
      labels: The associated bag labels of the input data.
      bag_class: String name of the desired bag class.
        The options are: "positive" or "negative".
      predictions: Class labels model predictions.
      If you don't specify anything, ground truth labels will be used.
      attention_weights: Attention weights for each instance within the input data.
      If you don't specify anything, the values won't be displayed.
    """
    return ##TODO
    labels = np.array(labels).reshape(-1)
    
    if bag_class =="positive":
        if predictions is not None:
            labels = np.where(predictions.argmax(1)==1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
            
        else:
            labels = np.where(labels == 1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
            
    elif bag_class == "negative":
        if predictions is not None:
            labels = np.where(predictions.argmax(1)==0)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
            
        else:
            labels = np.where(labels == 1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
        
    else:
         print(f"There is no class {bag_class}")
         return
     
    print(f"The bag class label is {bag_class}")
    
    for i in range(PLOT_SIZE):
        figure = plt.figure(figsize=(8,8))
        print(f"Bag number:{labels[i]}")
        for j in range(BAG_SIZE):
            image = bags[j][i]
            figure.add_subplot(1, BAG_SIZE, j + 1)
            plt.grid(False)
            if attention_weights is not None:
                plt.title(np.around(attention_weights[labels[i]][j],2))
            plt.imshow(image)
        plt.show()
        
# Plot some of validation data bags per class.
plot(val_data, val_labels, "positive")
plot(val_data, val_labels, "negative")