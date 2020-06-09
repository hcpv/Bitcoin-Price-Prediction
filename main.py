import yaml
import os
from data import *
from models import *
from visualize import *
from utils import *
from sklearn.preprocessing import MinMaxScaler

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
dropout = config["train_val"]["dropout"]
batch_size = config["train_val"]["batch_size"]
epochs = config["train_val"]["epochs"]
model_path = config["train_val"]["model_path"]
model_name = config["train_val"]["model_name"]
output_path = config["train_val"]["output_path"]

dates, price = load_data(data_path, data)

train_dates, train_price, test_dates, test_price = split_data(dates, price, val_size)
line_plot(convert(dates), price, convert(test_dates), test_price,
          output_path[:-1]+'/train_val.png', 'training', 'testing')

scaler = MinMaxScaler()
scaler.fit(train_price)
train_price = scaler.transform(train_price)
test_price = scaler.transform(test_price)

X_train_price, y_train_price = transform_data(train_price, n_days_in, n_days_out)
X_test_price, y_test_price = transform_data(test_price, n_days_in, n_days_out)
model = lstm(hidden_units, n_days_in, 1, n_days_out)

model.compile(loss='mse', optimizer='adam')

print('Training started')

model.fit(X_train_price, y_train_price, batch_size=batch_size, epochs=epochs)

print('Training completed')

y_pred_price = model.predict(X_test_price)
y_pred_price = scaler.inverse_transform(y_pred_price)
y_test_price = scaler.inverse_transform(y_test_price)
X_test_dates, y_test_dates = transform_data(test_dates, n_days_in, n_days_out)
info = model_name + '_' + str(len(hidden_units)) + '_layers_in_' + \
    str(n_days_in) + '_out_' + str(n_days_out)
path = output_path[:-1] + '/' + info + '.png'
line_plot(convert(y_test_dates), y_test_price, convert(y_test_dates), y_pred_price, path,
          'Actual', 'Predicted')

err = error(y_test_price, y_pred_price)
with open(os.path.join(output_path, 'mse.txt'), 'a') as file:
    file.write(info + ":" + str(err) + '\n')
print('Root Mean Squared Error:'+str(err))
