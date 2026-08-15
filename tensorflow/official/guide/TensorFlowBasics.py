#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 22:45:04 2026

@author: singh
"""
import tensorflow as tf

x = tf.constant([[1., 2. , 3.],
                [4., 5. , 6. ]])

print(x)
print(x.shape)
print(x.dtype)
print()
print("x+x:\n",x+x)
print()
print("5*x:\n",5*x)
print()
print("x @ tf.transpose(x):\n",x @ tf.transpose(x))
print()
print("tf.concat([x, x, x], axis=0):\n",tf.concat([x, x, x], axis=0))
print()
print("tf.nn.softmax(x, axis=-1):\n",tf.nn.softmax(x, axis=-1))
print()
print("tf.reduce_sum(x): \n",tf.reduce_sum(x))
print()
print("tf.convert_to_tensor([1,2,3]): \n",tf.convert_to_tensor([1,2,3]))
print()
print("tf.reduce_sum([1, 2, 3]):\n",tf.reduce_sum([1, 2, 3]))
print()
if tf.config.list_physical_devices('GPU'):
    print("Tensorflow **IS** using the GPU")
else:
    print("Tensorflow **IS NOT** using the GPU")
    

print("-----Variables-----")
print()
var = tf.Variable([0.0, 0.0, 0.0])
var.assign([1, 2, 3])
print("var = tf.Variable([0.0, 0.0, 0.0]): \n",var)
print()
print("var.assign_add([1,1,1]): \n",var.assign_add([1,1,1]))
print()
print("var.assign_add([1,1,1]):\n",var.assign_add([1,1,1]))
print("-----Automatic differentiation-----")
print()

x = tf.Variable(1.0)

def f(x):
    y = x**2 + 2*x - 5
    return y

print("f(x): \n",f(x))
print()

with tf.GradientTape() as tape:
    y=f(x)
    
g_x = tape.gradient(y,x) #gx=dy/dx
print("g_x: \n",g_x)

print("-----Graphs and tf.function-----")
print()

@tf.function
def my_func(x):
    print('Tracing.\n')
    return tf.reduce_sum(x)


x= tf.constant([1, 2, 3])
print(my_func(x))
x = tf.constant([10, 9, 8])#On subsequent calls TensorFlow only executes the optimized graph, skipping any non-TensorFlow steps. 
print(my_func(x))
print()
print("A graph may not be reusable for inputs with a different signature (shape and dtype), so a new graph is generated instead:")
print()
x = tf.constant([10.0, 9.1, 8.2], dtype=tf.float32)
print(my_func(x))

print("-----Modules, layers, and models-----")
print()
print("save and restore the values of your variables using tf.train.Checkpoint.")
print("import and export the tf.Variable values and the tf.function graphs using tf.saved_model. ")
print()
class MyModule(tf.Module):
    def __init__(self, value):
        self.weight = tf.Variable(value)
        
    
    @tf.function    
    def multiply(self, x):
        return x * self.weight
    

mod = MyModule(3)
print("mod = MyModule(3)")
print("mod.multiply(tf.constant([1, 2,3])):\n",mod.multiply(tf.constant([1, 2,3])))
print()
print("Save the Module")
save_path='./saved'
tf.saved_model.save(mod,save_path)
print("reload the saved the Module")
reloaded = tf.saved_model.load(save_path)
print(reloaded.multiply(tf.constant([1,2, 3])))
print()
print("-----Training loops-----")
print()
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['figure.figsize'] = [9, 6]

x = tf.linspace(-2, 2, 201)
x = tf.cast(x, tf.float32)

def f(x):
    y = x**2 + 2*x -5
    return y

y = f(x) + tf.random.normal(shape=[201])

plt.plot(x.numpy(), y.numpy(), '.', label= 'Data')
plt.plot(x, f(x),label='Ground truth')
plt.legend()
plt.show() 
print()
print("Create a quadratic model with randomly initialized weights and a bias:")
print()

class Model(tf.Module):
    
    def __init__(self):
        # Randomly generate weight and bias terms
        rand_init = tf.random.uniform(shape=[3], minval=0., maxval=5., seed=22)
        # Initialize model parameters
        self.w_q = tf.Variable(rand_init[0])
        self.w_l = tf.Variable(rand_init[1])
        self.b = tf.Variable(rand_init[2])
        
    
    @tf.function    
    def __call__(self, x):
        # Quadratic Model : quadratic_weight * x^2 + linear_weight * x + bias
        return self.w_q * (x**2) + self.w_l * x + self.b
    

print("First, observe your model's performance before training:")
print()
quad_model = Model()

def plot_preds(x, y, f, model, title):
    plt.figure()
    plt.plot(x, y, '.', label='Data')
    plt.plot(x, f(x), label='Ground truth')
    plt.plot(x, model(x), label='Predictions')
    plt.title(title)
    plt.legend()
    plt.show()
    
    
plot_preds(x, y, f, quad_model, "Before training")

print("Now, define a loss for your model:")
print()

def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y))


batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(buffer_size=x.shape[0]).batch(batch_size)

# Set training parameters
epochs = 100
learning_rate = 0.01
losses = []

# Format training loop
for epoch in range(epochs):
    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            batch_loss = mse_loss(quad_model(x_batch), y_batch)
        # Update parameters with respect to the gradient calculations
        grads = tape.gradient(batch_loss, quad_model.variables)
        for g,v in zip(grads, quad_model.variables):
            v.assign_sub(learning_rate*g)
    # Keep track of model loss per epoch
    loss = mse_loss(quad_model(x), y)
    losses.append(loss)
    if epoch % 10 == 0:
        print(f'Mean squared error for step {epoch}: {loss.numpy():0.3f}')
    
# Plot model results
print("\n")
plt.plot(range(epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.title('MSE loss vs training iterations')
plt.show()

print()
print("Now, observe your model's performance after training:")
print()
plot_preds(x, y, f, quad_model, 'After training')
print()
print("implementations of common training utilities are available in the tf.keras module")
print()
new_model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.stack([x, x**2], axis=1)),
    tf.keras.layers.Dense(units=1, kernel_initializer=tf.random.normal)])

new_model.compile(
    loss = tf.keras.losses.MSE,
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    )

history = new_model.fit(x, y, epochs=100, batch_size=32, verbose=0)

new_model.save('./my_new_model.keras')

print()
print("Observe your Keras model's performance after training:")
print()
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel([0,max(plt.ylim())])
plt.ylabel('Loss [Mean Squared Error]')
plt.title('Keras training progress')

plot_preds(x, y, f, new_model, 'After Training: Keras')