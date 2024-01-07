import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
df = pd.read_csv("META.csv")
plt.figure()
lag_plot(df['Open'], lag=3)
plt.title('META Stock - Autocorrelation plot with lag = 3')
plt.show()
plt.plot(df["Date"], df["Close"])
plt.xticks(np.arange(0,252, 200), df['Date'][0:252:200])
plt.title("META stock price over time")
plt.xlabel("time")
plt.ylabel("price")
plt.show()
train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]
training_data = train_data['Close'].values
test_data = test_data['Close'].values
history = [x for x in training_data]
model_predictions = []
