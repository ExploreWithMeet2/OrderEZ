from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


SEQUENCE_LENGTH = 30
EPOCHS = 75
BATCH_SIZE = 32
TEST_SIZE = 0.2
EARLY_STOP_PATIENCE = 20


np.random.seed(42)


def f():
    dates = pd.date_range(start="2024-01-01", periods=200, freq="D")

    trend = np.linspace(100, 150, len(dates))
    seasonality = 8 * np.sin(np.linspace(0, 3 * np.pi, len(dates)))
    noise = np.random.normal(0, 3.0, len(dates))

    values = trend + seasonality + noise

    df = pd.DataFrame({"date": dates, "value": values})

    scaler = MinMaxScaler()
    df["value_scaled"] = scaler.fit_transform(df[["value"]])

    def create_sequences(data: np.ndarray, window_size: int):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i : i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    X, y = create_sequences(df["value_scaled"].values, SEQUENCE_LENGTH)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    split_index = int(len(X) * (1 - TEST_SIZE))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    def build_model(input_shape: Tuple[int, int]) -> Sequential:

        model = Sequential(
            [
                LSTM(64, return_sequences=True, input_shape=input_shape),
                BatchNormalization(),
                Dropout(0.2),
                LSTM(32),
                BatchNormalization(),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

        return model

    model = build_model((SEQUENCE_LENGTH, 1))

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
