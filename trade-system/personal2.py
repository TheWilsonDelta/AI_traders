import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


google = yf.Ticker("BTC-USD")
df = google.history(period='60d', interval="5m")

df = df[['Low']]
df.head()

df['date'] = pd.to_datetime(df.index).time
df.set_index('date', inplace=True)
df.head()

y = df['Low'].values
# The split point is the 10% of the dataframe length
offset = int(0.01*len(df))
y_train = y[:-offset]
y_test  = y[-offset:]


model = ARIMA(y_train, order=(5,0,1)).fit()
forecast = model.forecast(steps=30)[0]



plt.plot(range(0,len(y_train)),y_train, label='Train', linewidth=0.2)
# plt.plot(range(len(y_train),len(y)),y_test,label='Test', linewidth=0.2)
# plt.plot(len(y)+30,forecast,label='Forcast',color='red', marker=".")
plt.title("Yahoo Data")
plt.legend()
plt.savefig("graph.svg")
plt.show()

print(f'Real data for time 0: {y_train[len(y_train)-1]}')
print(f'Real data for time 1: {y_test[0]}')
print(f'Pred data for time 1: {forecast}')