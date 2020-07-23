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
naive_pred = []

dataLength = len(avg)
windowSize = 100
lenPredict = 1

for i in np.arange(dataLength - windowSize - lenPredict - 1):
    inputData = avg[i:i+windowSize]
    outputData = avg[i+windowSize:i+windowSize+lenPredict]
    inputs.append(inputData)
    outputs.append(outputData)
    naive_pred.append(inputData[-1])

inputs = np.array(inputs)
outputs = np.array(outputs)
naive_pred = np.array(naive_pred)

# Separate training & test
inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
    inputs, outputs, test_size=0.2, random_state=101)

# Model
model = tf.keras.models.load_model('prediction_' + symbol)

# Predict
train_predicts = model.predict(inputs_train)
test_predicts = model.predict(inputs_test)
predicts = model.predict(inputs)

# Plot predictions
plt.scatter(np.arange(len(predicts)), predicts, c='red', s=2)
plt.plot(np.arange(len(outputs)), outputs, c='blue')
# plt.scatter(np.arange(len(naive_pred)), naive_pred, c='green', s=2)

# Plot outputs vs predictions
# plt.scatter(outputs_train, train_predicts, c='blue', s=2)
# plt.scatter(outputs_test, test_predicts, c='red', s=2)
# plt.scatter(outputs, naive_pred, c='green', s=2)

rmse = np.sqrt(np.mean((naive_pred-outputs)**2))
print('RMSE Naive:', rmse)

rmseRNN = np.sqrt(np.mean((predicts-outputs)**2))
print('RMSE RNN:', rmseRNN)

plt.show()
