# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
## Name : LISIANA T
## Reg NO: 212222240053
## Date :  

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
# The pmdarima package needs to be installed first. Install it using:
# !pip install pmdarima
from pmdarima import auto_arima # Importing auto_arima after installing the package

date_range = pd.date_range(start='2020-01-01', periods=1000, freq='D')
temperature_values = np.random.randint(15, 30, size=len(date_range))
weather_df = pd.DataFrame({'date': date_range, 'temperature': temperature_values})
weather_df.to_csv('/content/rainfall.csv', index=False)

df = pd.read_csv('/content/rainfall.csv', parse_dates=['date'], index_col='date')
print(df.head())
print(df.describe())

plt.figure(figsize=(12, 6))
plt.plot(df['temperature'])
plt.title('Time Series Plot of Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.show()

def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] < 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

adf_test(df['temperature'])

plt.figure(figsize=(12, 6))
plt.plot(df['temperature'])
plt.title('Time Series Plot of Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.show()
plot_acf(df['temperature'], lags=30)
plot_pacf(df['temperature'], lags=30)
plt.show()

df['temperature_diff'] = df['temperature'].diff().dropna()
plt.figure(figsize=(12, 6))
plt.plot(df['temperature_diff'])
plt.title('Differenced Time Series')
plt.xlabel('Date')
plt.ylabel('Differenced Temperature')
plt.show()

adf_test(df['temperature_diff'].dropna())

# Assuming p, d, q are defined somewhere before this line
# If not, you need to determine appropriate values for them
p, d, q = 1, 1, 1  # Example values, replace with your own

model = ARIMA(df['temperature'], order=(p, d, q))
model_fit = model.fit()

print(model_fit.summary())

forecast = model_fit.forecast(steps=10)
print(forecast)

plt.figure(figsize=(12, 6))
plt.plot(df['temperature'], label='Original Data')
plt.plot(forecast.index, forecast, color='red', label='Forecast', marker='o')
plt.title('ARIMA Model Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()
auto_model = auto_arima(df['temperature'], seasonal=False, stepwise=True, trace=True)
print(auto_model.summary())


train_size = int(len(df) * 0.8)
train, test = df['temperature'][:train_size], df['temperature'][train_size:]

# Assuming p, d, q
```
### OUTPUT:

![image](https://github.com/user-attachments/assets/f77af9f6-3a13-4585-a431-26d2f8895d49)

![image](https://github.com/user-attachments/assets/a8fb01eb-b7e3-470b-8b21-7ba5158971b8)

![image](https://github.com/user-attachments/assets/2ee81920-7a23-4a7a-a566-11bc4d6196b1)

![image](https://github.com/user-attachments/assets/722dc731-aa60-48fe-810d-2f6c6d3d20f7)

![image](https://github.com/user-attachments/assets/5701076d-0818-4578-bed4-4a3b76ddc8e5)

![image](https://github.com/user-attachments/assets/39bac444-ce05-436b-89dc-5e0ff2f81790)


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
