import tensorflow as tf
import keras
from keras import layers

print("A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.")
print()
model = keras.Sequential(
    [
        layers.Dense(2,activation='relu', name='layer1'),
        layers.Dense(3,activation='relu', name='layer2'),
        layers.Dense(4,name='layer3')
    ]
)

print(model.layers)
print()

#call model on a test input
x = tf.ones((3, 3))
y = model(x)

print("len(model.layers): ",len(model.layers))
#model.pop()
#print("after model.pop() len(model.layers): ",len(model.layers))
print("----- Creating a Sequential model (pass layers to the constructor) -----")
model = keras.Sequential([
    layers.Dense(2,activation='relu'),
    layers.Dense(3,activation='relu'),
    layers.Dense(4)
])
print(model.layers)
print("---create a Sequential model incrementally via the add() method:---")
model = keras.Sequential()
model.add(layers.Dense(2,activation='relu')),
model.add(layers.Dense(3,activation='relu')),
model.add(layers.Dense(4))
print("len(model.layers):",len(model.layers))
#model.pop()
#print("len(model.layers):",len(model.layers))
print("---Sequential constructor accepts a name argument,---")
model = keras.Sequential(name="my_sequential")
model.add(layers.Dense(2,activation='relu')),
model.add(layers.Dense(3,activation='relu')),
model.add(layers.Dense(4))
print()
print("---Specifying the input shape in advance---")
layer = layers.Dense(3)
print("layer.weights :",layer.weights)
x=tf.ones((1,4))
y = layer(x)
print("layer.weights :\n",layer.weights)

print()
print("---The weights are created when the model first sees some input data:---")
model = keras.Sequential([
    layers.Dense(2,activation='relu'),
    layers.Dense(3,activation='relu'),
    layers.Dense(4)
])
print("Call the model on a test input")
X = tf.ones((1,4))
y = model(X)
print("Number of weights after calling the model:", len(model.weights))
print("model.summary(): \n ",model.summary())
print()
print("---start your model by passing an Input object to your model, so that it knows its input shape from the start:---")
model = keras.Sequential()
model.add(keras.Input(shape=(4,)))
model.add(layers.Dense(2,activation='relu'))

print("model.summary(): \n ",model.summary())
print("model.layers: \n ",model.layers)
print()
print("---A simple alternative is to just pass an input_shape argument to your first layer:---")
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu", input_shape=(4,)))
print("model.summary(): \n ",model.summary())
print()
print("-----A common debugging workflow: add() + summary()-----")
print()
model = keras.Sequential()
model.add(keras.Input(shape=(250, 250, 3)))
model.add(layers.Conv2D(32, 5, strides=2,activation='relu'))
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.MaxPooling2D(3))

print("model.summary(): \n",model.summary())

model.add(layers.GlobalMaxPooling2D())
model.add(layers.Dense(10))
print()
print("-----Feature extraction with a Sequential model-----")
print()
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=[layer.output for layer in initial_model.layers],
)

# Call feature extractor on test input.
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)
print("features: \n ",features)
print()
print("example that only extract features from one layer:")
print()
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=initial_model.get_layer(name="my_intermediate_layer").output
)
# Call feature extractor on test input.
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)
print("features: \n ",features)
print()
print("-----Transfer learning with a Sequential model-----")
print()
model = keras.Sequential(
    [
        keras.Input(shape=(784)),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(10)
    ]
)

model.load_weights()
# Freeze all layers except the last one.
for layer in model.layers[:-1]:
    layer.trainable = False
# Recompile and train (this will only update the weights of the last layer).
model.compile(...)
model.fit(...)
