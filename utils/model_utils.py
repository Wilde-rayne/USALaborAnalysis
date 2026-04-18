import numpy as np
import pandas as pd

# Heavy ML deps (tensorflow, scikit-learn, scipy) are imported lazily inside
# the functions that need them. This keeps module import cheap for callers
# that only need the pure-numpy utilities (create_sliding_windows,
# forecast_with_model) and lets the test suite skip installing TensorFlow.


def create_sliding_windows(series: np.ndarray, window_size: int = 12):
    X, Y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        Y.append(series[i + window_size])
    return np.array(X), np.array(Y)

def build_lstm_rnn(input_timesteps: int = 12):
    import tensorflow as tf  # noqa: PLC0415 — lazy import; see module docstring

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, activation='relu', input_shape=(input_timesteps, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_test_rnn(
    ts_df: pd.DataFrame,
    colname: str,
    window: int = 12,
    test_size: float = 0.25,
    epochs: int = 10,
    batch_size: int = 16,
):
    """
    Fit an LSTM on ts_df['date', colname], return:
      model, metrics dict, last_window array
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error  # noqa: PLC0415
    from scipy.stats import f_oneway  # noqa: PLC0415

    # Prepare series
    df = ts_df.dropna(subset=[colname]).set_index('date').asfreq('MS')
    df.interpolate(method='time', inplace=True)
    df.fillna(inplace=True)
    values = df[colname].values
    if len(values) < window + 2:
        raise ValueError("Not enough data to train.")

    # Build windows
    X, Y = create_sliding_windows(values, window)
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    X_train = X_train.reshape((-1, window, 1))
    X_test  = X_test.reshape((-1, window, 1))

    # Train
    model = build_lstm_rnn(window)
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=epochs, batch_size=batch_size, verbose=0
    )

    # Evaluate
    preds_test = model.predict(X_test, verbose=0).flatten()
    rmse = float(np.sqrt(mean_squared_error(Y_test, preds_test)))
    mae  = float(mean_absolute_error(Y_test, preds_test))

    # ANOVA on residuals by quarter to detect seasonality shifts
    # group residuals by quarter
    residuals = Y_test - preds_test
    quarters = pd.to_datetime(df.index[window + split : window + split + len(residuals)])
    groups = [residuals[quarters.quarter == q] for q in [1,2,3,4]]
    f_stat, p_val = f_oneway(*[g for g in groups if len(g) > 1])

    # Save the last window for forecasting
    last_window = values[-window:]

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "anova_f": float(f_stat),
        "anova_p": float(p_val),
        "history": history.history
    }
    return model, metrics, last_window

def forecast_with_model(model, last_window: np.ndarray, months: int):
    """
    Roll forward `months` steps from `last_window` using `model.predict`.
    """
    window = len(last_window)
    seq = last_window.copy()
    preds, dates = [], []
    for i in range(months):
        inp = seq[-window:].reshape((1, window, 1))
        yhat = model.predict(inp, verbose=0)[0][0]
        preds.append(float(yhat))
        seq = np.append(seq, yhat)
    return preds
