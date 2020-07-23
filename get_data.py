import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

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

plt.plot(avg)
plt.show()