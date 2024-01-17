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

train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]

training_data = train_data['Close'].values
testing_data = test_data['Close'].values  
history = [x for x in training_data]

model_predictions = []
N_test_observations = len(testing_data)

for time_point in range(N_test_observations):
    model = ARIMA(history, order=(3, 2, 1))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = testing_data[time_point]
    history.append(true_test_value)

MSE_error = mean_squared_error(testing_data, model_predictions)
print('Testing Mean Squared Error is {}'.format(MSE_error))

test_set_range = df[int(len(df)*0.7):].index
plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
plt.plot(test_set_range, testing_data, color='red', label='Actual Price')
plt.title('META Prices Prediction')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.xticks(np.arange(0, len(test_set_range), 100), df['Date'].iloc[int(len(df)*0.7):][::100])
plt.legend()
plt.show()
