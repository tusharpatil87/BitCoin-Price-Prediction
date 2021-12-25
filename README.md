# Goal of the Project : Price Prediction for Bitcoin:

# Now a days Bitcoin cryptocurrency is too much famous and due to that most of the people are crazy to invest in the Bitcoin, so they are looking at the Bitcoin as  a safe investments with maximum returns. Bitcoin has too much trending like  stock market. There are many factors which are affecting the price of Bitcoin, that's the reason we are looking to predict the future price of the Bitcoin with the help of Open source libraries like Python, Facebook Prophet.


import numpy as np
import pandas as pd
from fbprophet import Prophet

path = r"E:\Skills\Machine_Larning\DATA_SCIENCE_PROJECT\2021\BitCoin\BTC-USD.csv"

df = pd.read_csv(path)

df.head()

df = df[["Date", "Close"]]

df.head()

df.columns = ["ds", "y"]

df.head()

import fbprophet
print(fbprophet.__version__)

pro = Prophet()

pro.fit(df)

future = pro.make_future_dataframe(periods=365)
print(future)

df.shape


forecast = pro.predict(future)


forecast

forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(200)

from fbprophet.plot import plot
pro.plot(forecast, figsize=(10, 10))

from fbprophet.plot import plot


pro.plot(forecast, figsize=(20, 5))

# Conclusion from the Project results:

# The historical data which we used for this project is showing as trend about the  price of the Bitcoin is going towards the higher side which means in future there are higher chances of increase in price of the Bitcoin. Please note this data is just showing as study purpose please do not invest just on the basis of this conclusion.

