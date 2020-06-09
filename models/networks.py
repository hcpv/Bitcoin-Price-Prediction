from tensorflow.keras.layers import LSTM, Dropout, Dense, LeakyReLU, Input
from tensorflow.keras.models import Sequential


def lstm(hidden_units, time_steps, n_features, output_size):
    model = Sequential()
    model.add(Input(shape=(time_steps, n_features)))
    for units in hidden_units[:-1]:
        model.add(LSTM(units=units, return_sequences=True))
    model.add(LSTM(units=hidden_units[-1]))
    model.add(Dense(output_size))
    print(model.summary())

    return model
