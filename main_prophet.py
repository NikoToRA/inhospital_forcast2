import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib

# 今日の日付を取得
today = datetime.today()

# 翌日と7日後の日付を計算
start_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
end_date = (today + timedelta(days=7)).strftime('%Y-%m-%d')

# データの読み込み
data = pd.read_csv('ultimate_pickup_data.csv')

# 日付と入院数の列を指定
data = data.rename(columns={'date': 'ds'})
# 'y'カラムはすでに存在するので変換不要

# Prophetモデルの作成
model = Prophet()
model.fit(data)

# モデルを保存
joblib.dump(model, 'prophet_model.joblib')
print("Prophet model saved successfully")

# 未来の日付を予測（60日分）
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# 翌日から7日後までの予測結果を表示
filtered_forecast = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]
print(filtered_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# 予測結果のプロット
fig = model.plot(forecast)
plt.show()