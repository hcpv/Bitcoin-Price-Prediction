from tensorflow.keras.layers import LSTM, Dropout, Dense, LeakyReLU, Input
from tensorflow.keras.models import Sequential


def lstm(hidden_units, time_steps, n_features, output_size, dropout=0.2):
    model = Sequential()
    model.add(Input(shape=(time_steps, n_features)))
    for units in hidden_units[:-1]:
        model.add(LSTM(units=units, return_sequences=True))
    model.add(LSTM(units=hidden_units[-1]))
    model.add(Dropout(dropout))
    model.add(Dense(output_size))
    model.add(LeakyReLU())
    print(model.summary())

    return model
