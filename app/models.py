import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .database import get_training_data, add_model_metrics
from sklearn.ensemble import RandomForestRegressor
from .prediction_manager import PredictionManager
import jpholiday

class PredictionModel:
    def __init__(self):
        self.rf_model = None
        self.prophet_model = None
        self.inpatient_levels = {
            'very_high': {'min': 35, 'color': '#dc3545', 'label': '非常に多い'},  # 赤色
            'high': {'min': 30, 'color': '#ffc107', 'label': '多い'},  # 黄色
            'normal': {'min': 25, 'color': '#28a745', 'label': '標準'},  # 緑色
            'low': {'min': 21, 'color': '#17a2b8', 'label': '少ない'},  # 水色
            'very_low': {'min': 0, 'color': '#6c757d', 'label': '非常に少ない'}  # グレー
        }
        self.prediction_manager = PredictionManager()
        self.load_models()
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
            rf_model_path = os.path.join(os.path.dirname(__file__), 'data', 'model.joblib')
            print(f"モデルパス: {rf_model_path}")  # デバッグ出力
            
            if os.path.exists(rf_model_path):
                print("モデルファイルが存在します。読み込みを開始します。")  # デバッグ出力
                self.rf_model = joblib.load(rf_model_path)
                print("RandomForest model loaded successfully")
                # モデルの学習状態を確認
                if hasattr(self.rf_model, 'n_features_in_'):
                    print("モデルは学習済みです。")
                else:
                    print("警告: モデルは学習されていないようです。再学習を実行します。")
                    self.retrain_model()
            else:
                print("モデルファイルが存在しません。新規作成します。")  # デバッグ出力
                # モデルが存在しない場合は新規作成
                self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                print("New RandomForest model created")
                # モデルの学習を実行
                result = self.retrain_model()
                if result['status'] == 'success':
                    print("モデルの学習が成功しました。")
                    # 学習済みモデルを保存
                    os.makedirs(os.path.dirname(rf_model_path), exist_ok=True)
                    joblib.dump(self.rf_model, rf_model_path)
                    print(f"モデルを保存しました: {rf_model_path}")
                else:
                    print(f"モデルの学習に失敗しました: {result['message']}")
                    raise Exception(f"モデルの学習に失敗しました: {result['message']}")

            # Prophetモデルの初期化
            self.prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                holidays_prior_scale=10.0,
                seasonality_mode='additive',
                changepoint_range=0.8
            )
            
            # 祝日データの追加
            holidays_df = self._create_holidays_df()
            self.prophet_model.add_country_holidays(country_name='JP')
            
            # Prophetモデルの学習
            self._train_prophet_model()

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

    def _create_holidays_df(self):
        """祝日データフレームの作成"""
        holidays = []
        current_year = datetime.now().year
        for year in range(current_year - 1, current_year + 2):
            for month in range(1, 13):
                for day in range(1, 32):
                    try:
                        date = datetime(year, month, day)
                        if jpholiday.is_holiday(date.date()):
                            holidays.append({
                                'holiday': 'JP_Holiday',
                                'ds': date,
                                'lower_window': -1,
                                'upper_window': 1
                            })
                    except ValueError:
                        continue
        return pd.DataFrame(holidays)

    def _train_prophet_model(self):
        """Prophetモデルの学習"""
        try:
            # トレーニングデータの取得
            training_data = get_training_data()
            if not training_data:
                print("警告: トレーニングデータが存在しません。")
                return

            # データの前処理
            df = pd.DataFrame(training_data)
            df['date'] = pd.to_datetime(df['date'])
            
            # Prophet用のデータフレームを作成
            prophet_df = pd.DataFrame({
                'ds': df['date'],
                'y': df['y']
            })

            # モデルの学習
            self.prophet_model.fit(prophet_df)
            print("Prophet model trained successfully")

        except Exception as e:
            print(f"Error training Prophet model: {str(e)}")
            raise

    def create_test_data(self, date, total_outpatient, intro_outpatient, er_patients, bed_count):
        """予測用のテストデータを生成"""
        try:
            # 日付の型チェックとエラーメッセージの詳細化
            if not date:
                raise ValueError("日付が指定されていません")
            
            # 日付をdatetime型に変換（文字列でもdatetime型でも対応）
            try:
                date = pd.to_datetime(date)
            except Exception as e:
                raise ValueError(f"日付の形式が正しくありません: {str(e)}")
            
            previous_day = date - timedelta(days=1)
            
            # デバッグ出力
            print(f"処理する日付: {date}, 型: {type(date)}")
            
            # 曜日を1-hotエンコーディング
            weekday = date.dayofweek  # 0=月曜日, 6=日曜日
            mon = 1 if weekday == 0 else 0
            tue = 1 if weekday == 1 else 0
            wed = 1 if weekday == 2 else 0
            thu = 1 if weekday == 3 else 0
            fri = 1 if weekday == 4 else 0
            sat = 1 if weekday == 5 else 0
            sun = 1 if weekday == 6 else 0
            
            # 祝日判定
            public_holiday = 1 if jpholiday.is_holiday(date) else 0
            public_holiday_previous_day = 1 if jpholiday.is_holiday(previous_day) else 0
            
            # 特徴量の作成
            features = {
                'mon': mon,
                'tue': tue,
                'wed': wed,
                'thu': thu,
                'fri': fri,
                'sat': sat,
                'sun': sun,
                'public_holiday': public_holiday,
                'public_holiday_previous_day': public_holiday_previous_day,
                'total_outpatient': int(total_outpatient),
                'intro_outpatient': int(intro_outpatient),
                'er': int(er_patients),
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

    def predict_weekly(self, date, bed_count):
        """指定された日付から1週間の予測を実行する（Prophetモデル使用）"""
        try:
            # 日付をdatetime型に変換
            date = pd.to_datetime(date)
            
            # 1週間分の予測日付を生成
            future_dates = pd.DataFrame({
                'ds': pd.date_range(start=date, periods=7, freq='D')
            })
            
            # Prophetモデルで予測を実行
            forecast = self.prophet_model.predict(future_dates)
            
            # 予測結果を整形
            daily_predictions = forecast['yhat'].values.tolist()
            
            # 各予測値を保存
            for i, pred_date in enumerate(future_dates['ds']):
                date_str = pred_date.strftime('%Y-%m-%d')
                self.prediction_manager.save_prediction(date_str, daily_predictions[i])
            
            return {
                'status': 'success',
                'daily_prediction': daily_predictions[0],
                'weekly_predictions': daily_predictions
            }
            
        except Exception as e:
            print(f"Error in weekly prediction: {str(e)}")
            return {
                'status': 'error',
                'message': f'週間予測中にエラーが発生しました: {str(e)}'
            }

    def predict(self, date, total_outpatient, intro_outpatient, er_patients, bed_count):
        """指定された日付の新規入院患者数を予測する（RandomForestモデル使用）"""
        try:
            # 予測用データの作成
            test_data = self.create_test_data(
                date=date,
                total_outpatient=total_outpatient,
                intro_outpatient=intro_outpatient,
                er_patients=er_patients,
                bed_count=bed_count
            )

            if test_data is None:
                return {
                    'status': 'error',
                    'message': '予測用データの作成に失敗しました'
                }

            # 予測の実行
            predicted_value = self.rf_model.predict(test_data)[0]
            
            # 日付をフォーマット
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            
            # 予測結果を保存
            self.prediction_manager.save_prediction(date_str, predicted_value)
            
            # 予測結果の評価
            evaluation = self.evaluate_inpatient_level(predicted_value)

            return {
                'status': 'success',
                'prediction': float(predicted_value),
                'date': date_str,
                'evaluation': evaluation
            }

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                'status': 'error',
                'message': f'予測中にエラーが発生しました: {str(e)}'
            }

    def get_predictions(self):
        """全ての予測を取得"""
        return self.prediction_manager.get_all_predictions()

    def update_actual_value(self, date, actual_value):
        """実際の値を更新"""
        return self.prediction_manager.update_actual_value(date, actual_value)

    def calculate_metrics(self):
        """予測の精度メトリクスを計算"""
        return self.prediction_manager.calculate_metrics()

    def retrain_model(self):
        """モデルの再学習を実行"""
        try:
            # トレーニングデータの取得
            print("トレーニングデータの取得を開始します。")  # デバッグ出力
            training_data = get_training_data()
            if not training_data:
                print("警告: トレーニングデータが存在しません。")  # デバッグ出力
                return {
                    'status': 'error',
                    'message': 'トレーニングデータが存在しません'
                }
            print(f"トレーニングデータの件数: {len(training_data)}")  # デバッグ出力

            # データの前処理
            df = pd.DataFrame(training_data, columns=[
                'id', 'date', 'total_outpatient', 'intro_outpatient',
                'er', 'bed_count', 'y', 'created_at'  # カラム名を修正
            ])
            print("データフレームの基本情報:")  # デバッグ出力
            print(df.info())
            
            # 特徴量の作成
            df['date'] = pd.to_datetime(df['date'])
            df['weekday'] = df['date'].dt.dayofweek
            df['mon'] = (df['weekday'] == 0).astype(int)
            df['tue'] = (df['weekday'] == 1).astype(int)
            df['wed'] = (df['weekday'] == 2).astype(int)
            df['thu'] = (df['weekday'] == 3).astype(int)
            df['fri'] = (df['weekday'] == 4).astype(int)
            df['sat'] = (df['weekday'] == 5).astype(int)
            df['sun'] = (df['weekday'] == 6).astype(int)
            
            # 祝日判定
            df['public_holiday'] = df['date'].apply(lambda x: 1 if jpholiday.is_holiday(x.date()) else 0)
            df['public_holiday_previous_day'] = df['date'].apply(lambda x: 1 if jpholiday.is_holiday((x - timedelta(days=1)).date()) else 0)

            # 特徴量とターゲットの分離
            feature_columns = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
                             'public_holiday', 'public_holiday_previous_day',
                             'total_outpatient', 'intro_outpatient', 'er', 'bed_count']
            X = df[feature_columns]
            y = df['y']

            print("特徴量の基本情報:")  # デバッグ出力
            print(X.info())
            print("\nターゲット変数の基本情報:")  # デバッグ出力
            print(y.describe())

            # モデルの再学習
            print("モデルの学習を開始します。")  # デバッグ出力
            self.rf_model.fit(X, y)
            print("モデルの学習が完了しました。")  # デバッグ出力

            # モデルの保存
            rf_model_path = os.path.join(os.path.dirname(__file__), 'data', 'model.joblib')
            os.makedirs(os.path.dirname(rf_model_path), exist_ok=True)
            joblib.dump(self.rf_model, rf_model_path)
            print(f"モデルを保存しました: {rf_model_path}")  # デバッグ出力

            # 精度評価
            y_pred = self.rf_model.predict(X)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            print(f"モデルの評価指標:")  # デバッグ出力
            print(f"MSE: {mse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R2 Score: {r2:.4f}")

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
                'er', 'bed_count', 'y', 'created_at'  # カラム名を修正
            ])
            
            # 特徴量の作成
            df['date'] = pd.to_datetime(df['date'])
            df['weekday'] = df['date'].dt.dayofweek
            df['mon'] = (df['weekday'] == 0).astype(int)
            df['tue'] = (df['weekday'] == 1).astype(int)
            df['wed'] = (df['weekday'] == 2).astype(int)
            df['thu'] = (df['weekday'] == 3).astype(int)
            df['fri'] = (df['weekday'] == 4).astype(int)
            df['sat'] = (df['weekday'] == 5).astype(int)
            df['sun'] = (df['weekday'] == 6).astype(int)
            
            # 祝日判定
            df['public_holiday'] = df['date'].apply(lambda x: 1 if jpholiday.is_holiday(x.date()) else 0)
            df['public_holiday_previous_day'] = df['date'].apply(lambda x: 1 if jpholiday.is_holiday((x - timedelta(days=1)).date()) else 0)

            # 特徴量とターゲットの分離
            feature_columns = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
                             'public_holiday', 'public_holiday_previous_day',
                             'total_outpatient', 'intro_outpatient', 'er', 'bed_count']
            X = df[feature_columns]
            y = df['y']

            # 予測と評価
            y_pred = self.rf_model.predict(X)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            print(f"モデルの評価指標:")  # デバッグ出力
            print(f"MSE: {mse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R2 Score: {r2:.4f}")

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