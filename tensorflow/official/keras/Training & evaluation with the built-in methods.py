from fileinput import filename

import tensorflow as tf
import keras
from cryptography.hazmat.asn1.asn1 import sequence
from keras import layers
from keras._tf_keras import keras
from pandas.plotting._matplotlib import timeseries
from prometheus_client import metrics

print("-----API overview: a first end-to-end example-----")
print()
inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)
print("history.history:\n",history.history)
print()
print("We evaluate the model on the test data via evaluate()")
print()
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)
print()
print("-----The compile() method: specifying a loss, metrics, and an optimizer-----")
print()
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

def get_uncompiled_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred), axis=-1)


model = get_uncompiled_model()
model.compile(optimizer=keras.optimizers.Adam(), loss=custom_mean_squared_error)

# We need to one-hot encode the labels to use MSE
y_train_one_hot = tf.one_hot(y_train, depth=10)
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)
print()
print("---subclass the keras.losses.Loss class---")
@keras.saving.register_keras_serializable()
class CustomMSE(keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred), axis=-1)
        return mse + reg * self.regularization_factor

    def get_config(self):
        return {
            "regularization_factor": self.regularization_factor,
            "name": self.name,
        }


model = get_uncompiled_model()
model.compile(optimizer=keras.optimizers.Adam(), loss=CustomMSE())

y_train_one_hot = tf.one_hot(y_train, depth=10)
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)
print()
print("-----Custom metrics  subclassing the keras.metrics.Metric class-----")
@keras.saving.register_keras_serializable()
class CategoricalTruePositives(keras.metrics.Metric):
    def __init__(self, name="categorical_true_positives", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.cast(values, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)


model = get_uncompiled_model()
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[CategoricalTruePositives()],
)
model.fit(x_train, y_train, batch_size=64, epochs=3)

print()
print("---Demo LogisticEndpoint layer---")
print()

@keras.saving.register_keras_serializable()
class LogisticEndpoint(layers.Layer):
    def __init__(self, name= None):
        super().__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, targets, logits, sample_weights=None):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(targets, logits, sample_weight=sample_weights)
        self.add_loss(loss)

        # Return the inference-time prediction tensor (for `.predict()`).
        return tf.nn.softmax(logits)


print()
print("---use it in a model with two inputs (input data & targets)---")
print()
import numpy as np

inputs = keras.Input(shape=(3,), name="inputs")
targets = keras.Input(shape=(10,), name="targets")
logits = keras.layers.Dense(10)(inputs)
predictions = LogisticEndpoint(name="predictions")(targets, logits)

model = keras.Model(inputs=[inputs, targets], outputs=predictions)
model.compile(optimizer="adam")

data = {
    "inputs": np.random.random((3, 3)),
    "targets": np.random.random((3, 10)),
}

model.fit(data)

print()
print("---Automatically setting apart a validation holdout set---")
print("the argument validation_split allows you to automatically reserve part of your training data for validation.")
print()
model=get_compiled_model()
model.fit(x_train, y_train, batch_size=64, validation_steps=0.2, epochs=1)

print()
print("---Training & evaluation from tf.data Datasets---")
print()
model = get_compiled_model()

# First, let's create a training Dataset instance.
# For the sake of our example, we'll use the same MNIST data as before.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# Shuffle and slice the dataset.
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# Now we get a test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(64)

# Since the dataset already takes care of batching,
# we don't pass a `batch_size` argument.
model.fit(train_dataset, epochs=3)

# You can also evaluate or predict on a dataset.
print("Evaluate")
result = model.evaluate(test_dataset)
dict(zip(model.metrics_names, result))

print()
print("steps_per_epoch argument > how many training steps the model should run before moving on to the next epoch")
print()
model = get_compiled_model()

# Prepare the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# Only use the 100 batches per epoch (that's 64 * 100 samples)
model.fit(train_dataset, epochs=3, steps_per_epoch=100)
print()
print("---Using a validation dataset---")
print()
model = get_compiled_model()

# Prepare the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# Prepare the validation dataset
val_dataset=tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

model.fit(train_dataset, epochs=1, validation_data=val_dataset)
print()
print("validation_steps > many validation steps the model should run with the validation dataset")
print()
model = get_compiled_model()

# Prepare the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# Prepare the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

model.fit(
    train_dataset,
    epochs=1,
    validation_data=val_dataset,
    validation_steps=10
)

print()
print("-----Other input formats supported-----")
print("Using a keras.utils.Sequence object as input")
print()
from skimage.io import imread
from skimage.transform import resize
import numpy as np

class CIFAR10Sequence(keras.utils.Sequence):
    def __init__(self, filenames, labels, batch_size):
        self.filenames, self.labels = filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y  = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([
            resize(imread(filename), (200, 200))
            for filename in batch_x]), np.array(batch_y)


#sequence= CIFAR10Sequence(filenames, labels, batch_size=self.batch_size)
#model.fit(sequence, epochs=10)
print()
print("Using sample weighting and class weighting")
class_weight = {
0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    # Set weight "2" for class "5",
    # making this class 2x more important
    5: 2.0,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.0,
}
print("Fit with class weight")
model = get_compiled_model()
model.fit(x_train, y_train,class_weight=class_weight, batch_size=64, epochs=1)
print()
print("Sample weights")
sample_weight = np.ones(shape=(len(y_train)))
sample_weight[y_train == 5] = 2.0
print("Fit with sample weight")
model = get_compiled_model()
model.fit(x_train, y_train,sample_weight=sample_weight, batch_size=64, epochs=1)
print()
print("Here's a matching Dataset example:")
sample_weight = np.ones(shape=(len(y_train)))
sample_weight[y_train == 5] = 2.0
# Create a Dataset that includes sample weights
# (3rd element in the return tuple).
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, sample_weight))

# Shuffle and slice the dataset.
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

model = get_compiled_model()
model.fit(train_dataset, epochs=1)
print()
print("-----Passing data to multi-input, multi-output models-----")
image_input = keras.Input(shape=(32, 32, 3),name="img_input")
timeseries_input = keras.Input(shape=(None, 10),name="ts_input")

x1 = layers.Conv2D(3, 3)(image_input)
x1 = layers.GlobalMaxPooling2D()(x1)

x2 = layers.Conv1D(3, 3)(timeseries_input)
x2 = layers.GlobalMaxPooling1D()(x2)

x = layers.Concatenate(axis=-1)([x1, x2])

score_output = layers.Dense(1, name="score_output")(x)
class_output = layers.Dense(5, name="class_output")(x)

model = keras.Model(
    inputs=[image_input, timeseries_input], outputs=[score_output, class_output]
)

keras.utils.plot_model(model,  "multi_input_and_output_model.png", show_shapes=True)

print("specify different losses to different outputs, by passing the loss functions as a list")
print()
model.compile(
    optimizer=keras.optimizers.RMSprop(lr=0.001),
    loss=[keras.losses.MeanSquaredError(),keras.losses.CategoricalCrossentropy()]
)

model.compile(
    optimizer=keras.optimizers.RMSprop(lr=0.001),
    loss=[keras.losses.MeanSquaredError(),keras.losses.CategoricalCrossentropy()],
    metrics=[
        [
            keras.metrics.MeanAbsolutePercentageError(),
            keras.metrics.MeanAbsoluteError()
        ],
        [keras.metrics.CategoricalAccuracy()]
    ]
)
print(" specify per-output losses and metrics via a dict:")
model.compile(
    optimizer=keras.optimizers.RMSprop(lr=0.001),
    loss={
        "score_output": keras.losses.MeanSquaredError(),
        "class_output": keras.losses.CategoricalCrossentropy()
    },
    metrics={
        "score_output": [
            keras.metrics.MeanAbsolutePercentageError(),
            keras.metrics.MeanAbsoluteError()
        ],
        "class_output": [keras.metrics.CategoricalAccuracy()],
     }
)
print("\"score\" loss in our example, by giving to 2x the importance of the class loss")
model.compile(
    optimizer=keras.optimizers.RMSprop(lr=0.001),
    loss={
        "score_output": keras.losses.MeanSquaredError(),
        "class_output": keras.losses.CategoricalCrossentropy()
    },
metrics={
        "score_output": [
            keras.metrics.MeanAbsolutePercentageError(),
            keras.metrics.MeanAbsoluteError()
        ],
        "class_output": [keras.metrics.CategoricalAccuracy()],
     },
    loss_weights={"score_output": 2.0, "class_output": 1.0}
)
print("choose not to compute a loss for certain outputs > are meant for prediction but not for training ")
# List loss version
model.compile(
    optimizer=keras.optimizers.RMSprop(lr=0.001),
    loss=[None, keras.losses.CategoricalCrossentropy()],
)
# Or dict loss version
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={"class_output": keras.losses.CategoricalCrossentropy()},
)