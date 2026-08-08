import numpy as np
import pandas as pd
import seaborn as sns
import time

#from IntroductionToNeuralNetworks import prediction

start_time = time.time()
training_data = pd.read_csv('Google_Stock_Price_Train.csv')

print(training_data.shape)
print(training_data.head())
print(training_data.describe())

# Lets work on the open stock price only and take out the " open " stock column.
training_data = training_data.iloc[:,1:2]
print(training_data.shape)
print(training_data.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(training_data, color='green')
plt.ylabel('Stock Price')
plt.xlabel('Time')
plt.title('Google Stock Price')
plt.show()

# Normalize the training data between [0,1]
from sklearn.preprocessing import MinMaxScaler
#the fit method, when applied to the training dataset, learns the model parameters (for example, mean and standard deviation).
#We then need to apply the transform method on the training dataset to get the transformed (scaled) training dataset.
#We could also perform both of this step in one step by applying fit_transform on the training dataset.
mm = MinMaxScaler(feature_range=(0,1))
training_data_scaled = mm.fit_transform(training_data)
print(training_data_scaled.shape)

plt.figure(figsize=(10,5))
plt.plot(training_data_scaled)
plt.title('Google Stock Price Prediction')
plt.ylabel('Stock Price')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.show()

print("most important part of the model > Feature Scaling")
x_train = training_data_scaled[59:1257]
y_train = training_data_scaled[60:1258]

print(x_train.shape)
print(y_train.shape)

# reshaping
x_train = np.reshape(x_train, (1198, 1, 1))
print(x_train.shape)

import keras
from keras.models import Sequential #helps to create model, layer by layer.
from keras.layers import Dense, LSTM, Dropout
#The dense layer is fully connected layer, so all the neurons in a layer are connected to those in a next layer.
#The dropout drops connections of neurons from the dense layer to prevent overfitting. the neurons whose value falls under 0, will be removed.
#LSTM gates to control the memorizing process. For detailed information on LSTM, go through the link below.
''' https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47 '''

# Create model using LSTM, Dropout and Dense layer as an output layer.
#Initializing the RNN
regressor = Sequential()
regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
regressor.fit(x_train, y_train, epochs=100, batch_size=32)

test_data = pd.read_csv('Google_Stock_Price_Test.csv')
test_stock = test_data.iloc[:,1:2]
print(len(test_stock))

input_value = test_stock
input_value = mm.transform(input_value)
# perform the same process, converting a 2D array to 3D
input_value = np.reshape(input_value, (20, 1, 1))

prediction = regressor.predict(input_value)
prediction = mm.inverse_transform(prediction)

plt.rcParams['figure.figsize'] = (15, 8)
plt.plot(test_stock, color='red', label ='Real Stock')
plt.plot(prediction, color='green', label ='Predicted Stock')
plt.title('Final Stock Prediction')
plt.legend()
plt.show()
