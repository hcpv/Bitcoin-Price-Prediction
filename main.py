import pandas as pd
import h5py
import yaml
import os
import numpy as np
from data import *
from models import *
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dropout, Dense, LeakyReLU
from tensorflow.keras.models import Sequential


with open(os.path.join('config.yaml')) as stream:
    config = yaml.safe_load(stream)

if config["dataset"]["download_required"]:
    download_and_save_data(data_path, data, days)
data = config["dataset"]["data"]
days = config["dataset"]["days"]
data_path = config["dataset"]["data_path"]
val_size = config["train_val"]["val_size"]
n_days_in = config["train_val"]["n_days_in"]
n_days_out = config["train_val"]["n_days_out"]
hidden_units = config["train_val"]["hidden_units"]
dates, price = load_data(data_path, data)

scaler = MinMaxScaler()
price = scaler.fit_transform(price)

_, _, X_test_dates, y_test_dates = transform_split_data(dates, n_days_in, n_days_out, val_size)

X_train_price, y_train_price, X_test_price, y_test_price = transform_split_data(
    price, n_days_in, n_days_out, val_size)

model = lstm_shallow(hidden_units, n_days_in, 1, n_days_out)
print(model.summary())

model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])

model.fit(X_train_price, y_train_price, validation_data=(
    X_test_price, y_test_price), batch_size=32, epochs=1000)

model.save("saved_model\lstm.h5")
