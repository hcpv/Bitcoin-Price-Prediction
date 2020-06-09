import yfinance as yf
import h5py
import os
import numpy as np


def download_and_save_data(data_path, data, days):
    print('Downloading data')
    filename = data + '.h5'
    df = yf.download([data])
    df = df.filter(['Close'])
    df = df.dropna().tail(days)
    prices = []
    dates = []
    for date, price in df.to_dict()['Close'].items():
        dates.append(date.timestamp())
        prices.append(price)

    with h5py.File(os.path.join(data_path, filename), 'w') as file:
        file.create_dataset("Date", data=dates)
        file.create_dataset("Price", data=prices)


def load_data(data_path, data):
    print('Loading data')
    filename = data + '.h5'
    with h5py.File(os.path.join(data_path, filename), 'r') as file:
        dates = file["Date"]
        price = file["Price"]
        dates = np.reshape(np.array(dates), (-1, 1))
        price = np.reshape(np.array(price), (-1, 1))
        return dates, price


def split_data(X, y, val_size):
    X = np.array(X)
    y = np.array(y)
    size = X.shape[0]
    split = size - val_size * size
    split = int(split)
    X_train = X[:split, :]
    y_train = y[:split, :]
    X_test = X[split:, :]
    y_test = y[split:, :]
    return X_train, y_train, X_test, y_test


def transform_data(data, n_days_in, n_days_out):
    X, y = [], []
    for i in range(len(data)):
        in_end = i + n_days_in
        out_end = in_end + n_days_out

        if out_end > len(data):
            break

        data_x, data_y = data[i:in_end], data[in_end:out_end]

        X.append(data_x)
        y.append(data_y)

    X = np.array(X)
    y = np.array(y)[:, :, 0]
    return X, y
