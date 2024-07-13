import yfinance as yf
import pandas as pd

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf


def train_and_predict_next_n_days(ticker_symbol, n_days):
    # Veri çekme ve kaydetme
    ticker_data = yf.Ticker(ticker_symbol)
    hist_data = ticker_data.history(period='5y')
    csv_file_path = 'hisse_senedi_verileri.csv'
    hist_data.to_csv(csv_file_path)
    print(f"Hisse senedi verileri {csv_file_path} dosyasına kaydedildi.")
    
    # Veriyi işleme ve model oluşturma
    data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True)
    data = data[['Open', 'Close']]
    for i in range(1, n_days+1):
        data[f'Open_lag{i}'] = data['Open'].shift(-i)  # Bir sonraki günlerin açılış fiyatı
    data.dropna(inplace=True)
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X = data_scaled[:-n_days, :-n_days]  # Son n_days satırı hariç tüm satırlar ve tüm sütunlar (hedef değişken hariç)
    y = data_scaled[:-n_days, -n_days:]  # Son n_days sütun (hedef değişken)
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # LSTM için veriyi 3 boyutlu hale getirme
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dense(n_days)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, verbose=0)  # verbose=0: Eğitim ilerlemesini göstermez
    
    # Son n_days'in tahmin edilen açılış fiyatlarını döndürme
    last_row = data_scaled[-1, :-n_days].reshape(1, 1, -n_days)  # Son satırı LSTM için uygun formata getir
    predicted_next_open_scaled = model.predict(last_row)
    last_row_with_prediction = np.hstack((last_row.reshape(1, -1), predicted_next_open_scaled))
    predicted_next_open = scaler.inverse_transform(last_row_with_prediction)[:, -n_days:]
    
    return predicted_next_open.flatten()