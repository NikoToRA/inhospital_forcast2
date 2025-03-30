import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .database import get_training_data, add_model_metrics
from sklearn.ensemble import RandomForestRegressor

class PredictionModel:
    def __init__(self):
        self.rf_model = None
        self.prophet_model = None
        self.load_models()
        # 5段階評価の基準値を設定
        self.inpatient_levels = {
            'very_high': {'min': 30, 'color': 'danger', 'label': '多い'},
            'high': {'min': 25, 'color': 'warning', 'label': 'やや多い'},
            'normal': {'min': 20, 'color': 'success', 'label': '標準'},
            'low': {'min': 15, 'color': 'info', 'label': 'やや少ない'},
            'very_low': {'min': 0, 'color': 'secondary', 'label': '少ない'}
        }
        # 曜日名の定義を追加
        self.weekday_names = {
            0: '月曜日',
            1: '火曜日',
            2: '水曜日',
            3: '木曜日',
            4: '金曜日',
            5: '土曜日',
            6: '日曜日'
        }

    def load_models(self):
        """保存されたモデルを読み込む"""
        try:
            # RandomForestモデルのパスを設定
            rf_model_path = os.path.join(os.path.dirname(__file__), '..', 'fixed_rf_model.joblib')
            if os.path.exists(rf_model_path):
                self.rf_model = joblib.load(rf_model_path)
                print("RandomForest model loaded successfully")
            else:
                # モデルが存在しない場合は新規作成
                self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                print("New RandomForest model created")

            # Prophetモデルのパスを設定
            prophet_model_path = os.path.join(os.path.dirname(__file__), '..', 'prophet_model.joblib')
            if os.path.exists(prophet_model_path):
                self.prophet_model = joblib.load(prophet_model_path)
                print("Prophet model loaded successfully")
            else:
                print("Prophet model not found. Using RandomForest model for weekly predictions.")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

    def create_test_data(self, date, total_outpatient, intro_outpatient, er_patients, bed_count):
        """予測用のテストデータを生成"""
        try:
            # 日付をdatetime型に変換
            date = pd.to_datetime(date)
            
            # 曜日を1-hotエンコーディング
            weekday = date.dayofweek  # 0=月曜日, 6=日曜日
            mon = 1 if weekday == 0 else 0
            tue = 1 if weekday == 1 else 0
            wed = 1 if weekday == 2 else 0
            thu = 1 if weekday == 3 else 0
            fri = 1 if weekday == 4 else 0
            sat = 1 if weekday == 5 else 0
            sun = 1 if weekday == 6 else 0
            
            # 特徴量の作成
            features = {
                'mon': mon,
                'tue': tue,
                'wed': wed,
                'thu': thu,
                'fri': fri,
                'sat': sat,
                'sun': sun,
                'public_holiday': 0,  # 祝日判定は別途実装が必要
                'public_holiday_previous_day': 0,  # 前日の祝日判定も別途実装が必要
                'total_outpatient': int(total_outpatient),
                'intro_outpatient': int(intro_outpatient),
                'ER': int(er_patients),  # 注意: 'ER'が正しい特徴量名
                'bed_count': int(bed_count)
            }
            
            print(f"生成された特徴量: {features}")  # デバッグ出力
            
            # DataFrameに変換
            df = pd.DataFrame([features])
            return df
        except Exception as e:
            print(f"Error creating test data: {str(e)}")
            return None

    def evaluate_inpatient_level(self, count):
        """新規入院患者数を5段階評価する"""
        for level, criteria in self.inpatient_levels.items():
            if count >= criteria['min']:
                return {
                    'level': level,
                    'color': criteria['color'],
                    'label': criteria['label']
                }
        return {
            'level': 'very_low',
            'color': 'secondary',
            'label': '少ない'
        }

    def predict(self, date, total_outpatient, intro_outpatient, er_patients, bed_count):
        """予測を実行"""
        try:
            if self.rf_model is None:
                print("RandomForestモデルが読み込まれていません")
                return {
                    'status': 'error',
                    'message': 'RandomForestモデルが読み込まれていません'
                }
            
            # 日付をdatetime型に変換
            date_obj = pd.to_datetime(date)
            weekday = date_obj.dayofweek  # 0=月曜日, 6=日曜日
            
            # 入力値の妥当性チェックと調整
            typical_values = {
                0: {'total_outpatient': 600, 'intro_outpatient': 15, 'er_patients': 15},  # 月曜日
                1: {'total_outpatient': 550, 'intro_outpatient': 12, 'er_patients': 12},  # 火曜日
                2: {'total_outpatient': 650, 'intro_outpatient': 14, 'er_patients': 13},  # 水曜日
                3: {'total_outpatient': 500, 'intro_outpatient': 12, 'er_patients': 12},  # 木曜日
                4: {'total_outpatient': 600, 'intro_outpatient': 13, 'er_patients': 14},  # 金曜日
                5: {'total_outpatient': 200, 'intro_outpatient': 5, 'er_patients': 15},   # 土曜日
                6: {'total_outpatient': 30, 'intro_outpatient': 2, 'er_patients': 12}     # 日曜日
            }
            
            # 入力値が通常範囲から大きく外れている場合の警告
            typical = typical_values[weekday]
            warnings = []
            if total_outpatient < typical['total_outpatient'] * 0.3:
                warnings.append(f"外来患者数が{self.weekday_names[weekday]}の通常値（{typical['total_outpatient']}人程度）より大幅に少なくなっています")
            if intro_outpatient < typical['intro_outpatient'] * 0.3:
                warnings.append(f"紹介患者数が{self.weekday_names[weekday]}の通常値（{typical['intro_outpatient']}人程度）より大幅に少なくなっています")
            
            # テストデータの生成
            test_data = self.create_test_data(date, total_outpatient, intro_outpatient, er_patients, bed_count)
            if test_data is None:
                return {
                    'status': 'error',
                    'message': 'テストデータの生成に失敗しました'
                }
            
            # RandomForestモデルで当日の予測を実行
            daily_prediction = self.rf_model.predict(test_data)[0]
            print(f"当日予測値（RandomForest）: {daily_prediction}")
            print(f"入力値 - 外来患者数: {total_outpatient}, 紹介患者数: {intro_outpatient}, ER患者数: {er_patients}, ベッド数: {bed_count}")
            
            # 予測値の調整（極端に低い入力値の場合）
            if warnings and weekday in [0, 1, 2, 3, 4]:  # 平日の場合
                adjustment_factor = (total_outpatient / typical['total_outpatient'] + 
                                  intro_outpatient / typical['intro_outpatient']) / 2
                if adjustment_factor < 0.3:  # 入力値が通常の30%未満
                    daily_prediction = daily_prediction * (1 + (0.3 - adjustment_factor))
            
            # 当日の予測値を5段階評価
            evaluation = self.evaluate_inpatient_level(daily_prediction)
            
            # Prophetモデルで週間予測を実行
            weekly_predictions = []
            base_date = pd.to_datetime(date)
            
            if self.prophet_model is not None:
                # Prophetモデルを使用した週間予測
                future_dates = pd.DataFrame({
                    'ds': [base_date + timedelta(days=i) for i in range(7)]
                })
                prophet_forecast = self.prophet_model.predict(future_dates)
                
                # 週間予測の結果を処理
                for i in range(7):
                    current_date = base_date + timedelta(days=i)
                    # Prophetの予測値を取得
                    prophet_pred = prophet_forecast.iloc[i]['yhat']
                    prophet_lower = prophet_forecast.iloc[i]['yhat_lower']
                    prophet_upper = prophet_forecast.iloc[i]['yhat_upper']
                    
                    # 予測値が現実的な範囲内に収まるように調整
                    prophet_pred = max(0, min(prophet_pred, bed_count))
                    prophet_lower = max(0, min(prophet_lower, bed_count))
                    prophet_upper = max(0, min(prophet_upper, bed_count))
                    
                    evaluation_weekly = self.evaluate_inpatient_level(prophet_pred)
                    weekly_predictions.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'predicted_inpatients': int(prophet_pred),
                        'lower_bound': int(prophet_lower),
                        'upper_bound': int(prophet_upper),
                        'level': evaluation_weekly['level'],
                        'color': evaluation_weekly['color'],
                        'label': evaluation_weekly['label']
                    })
            
            return {
                'status': 'success',
                'daily_prediction': int(daily_prediction),
                'evaluation': evaluation,
                'weekly_predictions': weekly_predictions,
                'warnings': warnings if warnings else None
            }
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def retrain_model(self):
        """モデルの再学習を実行"""
        try:
            # トレーニングデータの取得
            training_data = get_training_data()
            if not training_data:
                return {
                    'status': 'error',
                    'message': 'トレーニングデータが存在しません'
                }

            # データの前処理
            df = pd.DataFrame(training_data, columns=[
                'id', 'date', 'total_outpatient', 'intro_outpatient',
                'er_patients', 'bed_count', 'actual_inpatients', 'created_at'
            ])
            
            # 特徴量の作成
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            df['is_holiday'] = 0  # 休日判定は別途実装が必要

            # 特徴量とターゲットの分離
            X = df[['total_outpatient', 'intro_outpatient', 'er_patients', 'bed_count',
                   'day_of_week', 'month', 'year', 'is_holiday']]
            y = df['actual_inpatients']

            # モデルの再学習
            self.rf_model.fit(X, y)

            # モデルの保存
            rf_model_path = os.path.join(os.path.dirname(__file__), '..', 'fixed_rf_model.joblib')
            joblib.dump(self.rf_model, rf_model_path)

            # 精度評価
            y_pred = self.rf_model.predict(X)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # 評価指標の保存
            add_model_metrics('RandomForest', mse, mae, r2)

            return {
                'status': 'success',
                'message': 'モデルの再学習が完了しました',
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2_score': r2
                }
            }
        except Exception as e:
            print(f"Error in model retraining: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def evaluate_model(self):
        """モデルの精度評価を実行"""
        try:
            # トレーニングデータの取得
            training_data = get_training_data()
            if not training_data:
                return {
                    'status': 'error',
                    'message': 'トレーニングデータが存在しません'
                }

            # データの前処理
            df = pd.DataFrame(training_data, columns=[
                'id', 'date', 'total_outpatient', 'intro_outpatient',
                'er_patients', 'bed_count', 'actual_inpatients', 'created_at'
            ])
            
            # 特徴量の作成
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            df['is_holiday'] = 0  # 休日判定は別途実装が必要

            # 特徴量とターゲットの分離
            X = df[['total_outpatient', 'intro_outpatient', 'er_patients', 'bed_count',
                   'day_of_week', 'month', 'year', 'is_holiday']]
            y = df['actual_inpatients']

            # 予測と評価
            y_pred = self.rf_model.predict(X)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # 評価指標の保存
            add_model_metrics('RandomForest', mse, mae, r2)

            return {
                'status': 'success',
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2_score': r2
                }
            }
        except Exception as e:
            print(f"Error in model evaluation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            } 