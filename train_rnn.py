import yfinance as yf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime

symbol = 'IBM'
stockData = yf.Ticker(symbol)

# Get stock info
data = stockData.history(period="5y")

# Get close price
close = data['Close'].to_numpy()

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

avg = moving_average(close, 10)

# Normalise
avg = (avg - avg.min()) / (avg.max() - avg.min())

# Prepare
inputs = []
outputs = []

dataLength = len(avg)
windowSize = 100
lenPredict = 1

for i in np.arange(dataLength - windowSize - lenPredict - 1):
    inputData = avg[i:i+windowSize]
    outputData = avg[i+windowSize:i+windowSize+lenPredict]
    inputs.append(inputData)
    outputs.append(outputData)

inputs = np.array(inputs)
outputs = np.array(outputs)

# Separate training & test
inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
    inputs, outputs, test_size=0.2, random_state=101)

# Paramaters
input_neurons = windowSize
rnn_cells = 10
rnn_input_shape = [windowSize, 1]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(input_neurons, input_shape=[windowSize]),
    tf.keras.layers.Reshape(rnn_input_shape),
    tf.keras.layers.LSTM(rnn_cells, activation='relu'),
    tf.keras.layers.Dense(lenPredict)
])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

# Compile
model.compile(optimizer='adam', loss='mse')

# Fit
model.fit(inputs_train, outputs_train, epochs=1000,
          validation_split=0.1, callbacks=[tensorboard_callback])

# Eval
model.evaluate(inputs_test, outputs_test, verbose=2)

# Save
model.save('prediction_' + symbol)