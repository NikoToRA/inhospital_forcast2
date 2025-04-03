import pandas as pd
import numpy as np
import jpholiday
from datetime import datetime, timedelta
import joblib
import os
from .database import save_prediction, get_predictions, get_prediction_for_date, get_training_data, add_model_metrics

class PredictionManager:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), 'data', 'model.joblib')
        self.predictions_df = self._load_predictions()
        self.training_data = self._load_training_data()
        self.model = self._load_model()

    def _load_predictions(self):
        """予測データを読み込む"""
        predictions = get_predictions()
        if predictions:
            return pd.DataFrame(predictions)
        return pd.DataFrame(columns=['date', 'predicted_value', 'actual_value', 'created_at'])

    def _load_training_data(self):
        """トレーニングデータを読み込む"""
        data = get_training_data()
        if data:
            return pd.DataFrame(data)
        return pd.DataFrame(columns=['date', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
                                   'public_holiday', 'public_holiday_previous_day',
                                   'total_outpatient', 'intro_outpatient', 'er',
                                   'bed_count', 'y'])

    def _load_model(self):
        """保存されたモデルを読み込む"""
        if os.path.exists(self.model_path):
            try:
                return joblib.load(self.model_path)
            except Exception as e:
                print(f"Warning: Failed to load model from {self.model_path}: {e}")
                return None
        return None

    def save_model(self, model):
        """モデルを保存する"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(model, self.model_path)
            self.model = model
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def prepare_features(self, date):
        """予測用の特徴量を準備する"""
        date_dt = pd.to_datetime(date)
        
        # 曜日フラグの設定
        features = {
            'mon': date_dt.weekday() == 0,
            'tue': date_dt.weekday() == 1,
            'wed': date_dt.weekday() == 2,
            'thu': date_dt.weekday() == 3,
            'fri': date_dt.weekday() == 4,
            'sat': date_dt.weekday() == 5,
            'sun': date_dt.weekday() == 6,
            'public_holiday': jpholiday.is_holiday(date_dt.date()),
            'public_holiday_previous_day': jpholiday.is_holiday((date_dt - timedelta(days=1)).date())
        }

        # 過去の予測データから特徴量を追加
        past_predictions = self.predictions_df[self.predictions_df['date'] < date]
        if not past_predictions.empty:
            # 過去7日間の予測値の平均
            last_7_days = past_predictions.tail(7)
            features['predicted_value_7d_avg'] = last_7_days['predicted_value'].mean()
            
            # 過去7日間の予測値の標準偏差
            features['predicted_value_7d_std'] = last_7_days['predicted_value'].std()
            
            # 過去7日間の予測値の最大値
            features['predicted_value_7d_max'] = last_7_days['predicted_value'].max()
            
            # 過去7日間の予測値の最小値
            features['predicted_value_7d_min'] = last_7_days['predicted_value'].min()

        # トレーニングデータから特徴量を追加
        past_training = self.training_data[self.training_data['date'] < date]
        if not past_training.empty:
            # 過去7日間の実際の値の平均
            last_7_days = past_training.tail(7)
            features['actual_value_7d_avg'] = last_7_days['y'].mean()
            
            # 過去7日間の実際の値の標準偏差
            features['actual_value_7d_std'] = last_7_days['y'].std()
            
            # 過去7日間の実際の値の最大値
            features['actual_value_7d_max'] = last_7_days['y'].max()
            
            # 過去7日間の実際の値の最小値
            features['actual_value_7d_min'] = last_7_days['y'].min()

        return pd.DataFrame([features])

    def predict(self, date):
        """指定された日付の予測を行う"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train and save a model first.")

        features = self.prepare_features(date)
        predicted_value = self.model.predict(features)[0]
        
        # 予測結果を保存
        self.save_prediction(date, predicted_value)
        
        return predicted_value

    def save_prediction(self, date, predicted_value, actual_value=None):
        """新しい予測を保存する"""
        # 新しい予測データを作成
        new_prediction = {
            'date': date,
            'predicted_value': predicted_value,
            'actual_value': actual_value,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Supabaseに保存
        saved_data = save_prediction(date, predicted_value, actual_value)
        if saved_data:
            # 同じ日付のデータがある場合は削除
            self.predictions_df = self.predictions_df[self.predictions_df['date'] != date]
            
            # 新しい予測を追加
            new_df = pd.DataFrame([new_prediction])
            self.predictions_df = pd.concat([self.predictions_df, new_df], ignore_index=True)
            return True
        return False

    def get_all_predictions(self):
        """全ての予測データを取得"""
        return self.predictions_df

    def get_prediction_for_date(self, date):
        """特定の日付の予測を取得"""
        prediction = get_prediction_for_date(date)
        if prediction:
            return pd.Series(prediction)
        return None

    def calculate_metrics(self):
        """予測の精度メトリクスを計算"""
        df = self.predictions_df.dropna(subset=['actual_value'])
        if len(df) == 0:
            return None

        mse = ((df['predicted_value'] - df['actual_value']) ** 2).mean()
        mae = abs(df['predicted_value'] - df['actual_value']).mean()
        
        # メトリクスを保存
        add_model_metrics(
            date=datetime.now().strftime('%Y-%m-%d'),
            mse=mse,
            mae=mae,
            prediction_count=len(df)
        )
        
        return {
            'mse': mse,
            'mae': mae,
            'count': len(df)
        }

    def get_data_for_date(self, date):
        """指定された日付のデータを取得"""
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        return self.training_data[self.training_data['date'] == date_str].to_dict('records')[0] if not self.training_data[self.training_data['date'] == date_str].empty else None

    def update_actual_value(self, date, actual_value):
        """実際の値を更新する"""
        try:
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            if not self.predictions_df[self.predictions_df['date'] == date_str].empty:
                self.predictions_df.loc[self.predictions_df['date'] == date_str, 'actual_value'] = float(actual_value)
                return self.save_prediction(date_str, self.predictions_df.loc[self.predictions_df['date'] == date_str, 'predicted_value'].iloc[0], actual_value)
            return False
        except Exception as e:
            print(f"Error updating actual value: {str(e)}")
            return False 