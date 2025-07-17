# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from vnstock import Vnstock # Nạp thư viện để sử dụng
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# ⚙️ Hàm dự đoán giá với LSTM
def predict_next_days(df_close, future_days):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_close.values.reshape(-1, 1))

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    if len(X) < 1:
        return None
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    last_60 = scaled[-60:]
    preds = []
    for _ in range(future_days):
        input_seq = last_60.reshape(1, 60, 1)
        pred = model.predict(input_seq, verbose=0)[0][0]
        preds.append(pred)
        last_60 = np.append(last_60[1:], [[pred]], axis=0)

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1))

# Giao diện Streamlit
st.set_page_config(page_title="Dự đoán giá cổ phiếu", layout="wide")
st.title("📈 Dự đoán giá cổ phiếu Việt Nam theo ngày")

symbols = ['TCB','VES','VIC','VHM','FPT','FRT','VPB','HPG','SHB','CTR','MWG','CTD','NVL']
selected = st.multiselect("📌 Chọn mã cổ phiếu", symbols, default=['HPG', 'FPT'])
n_days = st.radio("📆 Số ngày dự đoán", [5, 7, 10])

if st.button("🚀 Dự đoán"):
    start_date = '2024-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    col1, col2 = st.columns(2)

    for i, stock_name in enumerate(selected):
        try:
            
            stock = Vnstock().stock(symbol=stock_name, source='VCI') # Định nghĩa biến vnstock lưu thông tin mã chứng khoán & nguồn dữ liệu bạn sử dụng
            df = stock.quote.history(start='2023-01-01', end=str(end_date), interval='1D') # Thiết lập thời gian tải dữ liệu và khung thời gian tra cứu là 1 ngày

            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            forecast = predict_next_days(df['close'], n_days)

            if forecast is not None:
                last_day = df.index[-1]
                future_dates = [last_day + timedelta(days=x+1) for x in range(n_days)]
                forecast_df = pd.DataFrame({
                    'Ngày': future_dates,
                    'Giá dự đoán': forecast.flatten()
                }).set_index('Ngày')

                with col1 if i % 2 == 0 else col2:
                    st.subheader(f"🔮 {stock_name} – Dự đoán {n_days} ngày tới")
                    st.line_chart(forecast_df)
                    st.dataframe(forecast_df.style.format("{:,.2f}"))
            else:
                st.warning(f"❕ Không đủ dữ liệu để dự đoán cho {stock_name}")
        except Exception as e:
            st.error(f"❌ Lỗi với {stock_name}: {e}")
