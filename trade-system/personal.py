import datetime as dt
import json
from time import sleep

import pandas as pd
import requests
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


def get_finance_data():
    # print('Getting data from Binance...')

    url = 'https://api.binance.com/api/v3/klines'
    symbol = 'BTCUSDT'
    interval = '1m'
    start = str(int(dt.datetime(2022, 1, 1).timestamp() * 1000))
    par = {'symbol': symbol, 'interval': interval, 'startTime': start}
    df = pd.DataFrame(json.loads(requests.get(url, params=par).text))
    # format columns name
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
                  'taker_base_vol', 'taker_quote_vol', 'ignore']
    df.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in df.datetime]
    df = df.astype(float)

    df = df[['low']]
    df['datetime'] = pd.to_datetime(df.index).time
    df.set_index('datetime', inplace=True)

    print(f'Data found. Last value dated on {df.index[-1]}')

    y = df['low'].values
    model = ARIMA(y, order=(5, 0, 1)).fit()
    forecast = model.forecast(steps=60)[0]

    df.head()
    y = df['low'].values
    X = df.index.values
    # The split point is the 10% of the dataframe length
    offset = int(0.10 * len(df))
    y_train = y[:-offset]
    y_test = y[-offset:]
    X_test = X[-offset:]
    X_train = X[:-offset]
    plt.plot(range(0, len(y_train)), y_train, label='Train')
    plt.plot(range(len(y_train), len(y)), y_test, label='Test')
    plt.plot(len(y) + 60, forecast, label='Forcast', color='red', marker=".")
    plt.title("Binance Data")
    plt.xlabel("Date/Time")
    plt.legend()
    plt.grid()
    plt.show()
    model = ARIMA(y_train, order=(5, 0, 1)).fit()
    forecast = model.forecast(steps=60)[0]
    print(f'Real data for time 0: {y_train[len(y_train) - 1]}')
    print(f'Real data for time 1: {y_test[0]}')
    print(f'Pred data for time 1: {forecast}')

    return df


def get_forecast():
    df = get_finance_data()

    # Assuming that we've properly trained the model before and that the
    # hyperparameters are correctly tweaked, we use the full dataset to fit
    y = df['low'].values
    model = ARIMA(y, order=(5, 0, 1)).fit()
    forecast = model.forecast(steps=60)[0]

    # Returning the last real data and the forecast for the next minute
    print(f"Forcast is: {y[len(y) - 1], forecast}")
    return (y[len(y) - 1], forecast)


i = 1
holding_token = 0
purchase_price = 0
total_profit = 0

print("AI trading system starting...")
print("===============================")

while i == 1:
    (last_real_data, forecast) = get_forecast()
    print("===============================")

    if (forecast > last_real_data) and holding_token == 0:
        purchase_price = last_real_data
        print(f'Bought 1 token for {last_real_data}')
        print("===============================")
        holding_token = 1
    elif (forecast < last_real_data) and holding_token == 1:
        total_profit = last_real_data - purchase_price
        print(f'Sold 1 token purchased for {purchase_price} at {last_real_data}')
        print(
            f'You have made a profit/loss of: {total_profit} [{(purchase_price/last_real_data)}%]')
        print("===============================")
        holding_token = 0
    elif holding_token == 1:
        print(f'Currently holding a token')
        print("===============================")
    else:
        print(f'No tokens were bought nor sold. Current profit is {total_profit}')
        print("===============================")



    sleep(60)
