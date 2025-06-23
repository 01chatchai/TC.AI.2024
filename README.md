# TC.AI.2024
ขั้นตอนการทำงาน
1.	เตรียมข้อมูลพายุในอดีต (เช่น จาก IBTrACS หรือ JTWC)
2.	ใช้ข้อมูลลำดับเวลา (Latitude, Longitude) และตัวแปรอุตุนิยมวิทยา
3.	สร้างโมเดล LSTM
4.	พยากรณ์เส้นทางพายุในอนาคต
ตัวอย่างโค้ด
1. ติดตั้งไลบรารี
python
คัดลอกแก้ไข
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

2. เตรียมข้อมูล
สมมติว่าข้อมูลพายุในอดีตอยู่ในรูปแบบ .csv ที่มีคอลัมน์ Latitude, Longitude, WindSpeed, และ Pressure
python
คัดลอกแก้ไข
# โหลดข้อมูลพายุ
data = pd.read_csv("tropical_cyclone_tracks.csv")

# เลือกฟีเจอร์ที่สำคัญ
features = ['Latitude', 'Longitude', 'WindSpeed', 'Pressure']
data = data[features]

# การปรับขนาดข้อมูล (Normalization)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# สร้างลำดับข้อมูล (Time Series)
sequence_length = 10  # ใช้ข้อมูล 10 จุดเวลา
X, y = [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i+sequence_length])
    y.append(data_scaled[i+sequence_length])

X = np.array(X)
y = np.array(y)

print("Shape of X:", X.shape)  # (จำนวนตัวอย่าง, ลำดับเวลา, จำนวนฟีเจอร์)
print("Shape of y:", y.shape)  # (จำนวนตัวอย่าง, จำนวนฟีเจอร์)

3. สร้างโมเดล LSTM
python
คัดลอกแก้ไข
# สร้างโมเดล LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    LSTM(32),
    Dense(y.shape[1])
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

4. ฝึกโมเดล
python
คัดลอกแก้ไข
# แบ่งข้อมูล Train และ Test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ฝึกโมเดล
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# แสดงกราฟ Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

5. ทำนายเส้นทางพายุ
python
คัดลอกแก้ไข
# ทำนายข้อมูล Test
predictions = model.predict(X_test)

# แปลงค่ากลับสู่ช่วงเดิม
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)

# แสดงผลการพยากรณ์ (Latitude และ Longitude)
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled[:, 0], y_test_rescaled[:, 1], label='Actual Path', color='blue')
plt.plot(predictions_rescaled[:, 0], predictions_rescaled[:, 1], label='Predicted Path', color='red', linestyle='dashed')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend()
plt.title("Tropical Cyclone Track Prediction")
plt.show()

การปรับปรุงเพิ่มเติม
1.	เพิ่มตัวแปร: เช่น Sea Surface Temperature (SST), Relative Humidity
2.	ผสมผสาน CNN และ LSTM: เพื่อรวมข้อมูลภาพดาวเทียม
3.	ปรับพารามิเตอร์: เช่น จำนวนชั้น LSTM, Learning Rate

