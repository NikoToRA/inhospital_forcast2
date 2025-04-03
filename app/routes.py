from flask import Blueprint, render_template, request, jsonify
from datetime import datetime, timedelta
from app.models import PredictionModel
import pandas as pd
import numpy as np
import requests
from dateutil.relativedelta import relativedelta
import calendar
import os
import joblib
from prophet import Prophet
import jpholiday
from .database import get_training_data, add_model_metrics, save_input_data

main = Blueprint('main', __name__)

# 予測モデルのインスタンスを作成
prediction_model = PredictionModel()

def get_japanese_holidays(year):
    """日本の祝日を取得"""
    holidays = {}
    for holiday in jpholiday.year_holidays(year):
        date_str = holiday[0].strftime('%Y-%m-%d')
        name = holiday[1]
        holidays[date_str] = name
    return holidays

def get_default_values(date):
    """日付に基づいてデフォルト値を設定"""
    date_obj = pd.to_datetime(date)
    month = date_obj.month
    weekday = date_obj.dayofweek
    
    # 曜日による調整
    if weekday in [5, 6]:  # 土日
        total_outpatient = 400  # 平日より少ない
        er_patients = 15       # 平日より少ない
    else:  # 平日
        # 季節による調整
        if month in [12, 1, 2]:  # 冬季
            total_outpatient = 720  # 600 * 1.2
            er_patients = 26       # 20 * 1.3
        elif month in [6, 7, 8]:  # 夏季
            total_outpatient = 540  # 600 * 0.9
            er_patients = 22       # 20 * 1.1
        else:  # 春秋
            total_outpatient = 600
            er_patients = 20
    
    return {
        'total_outpatient': int(total_outpatient),
        'intro_outpatient': 30,
        'er_patients': er_patients,
        'bed_count': 300
    }

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/monthly_calendar')
def monthly_calendar():
    today = datetime.now()
    year = request.args.get('year', type=int, default=today.year)
    month = request.args.get('month', type=int, default=today.month)
    
    # 月の最初の日と最後の日を取得
    first_day = datetime(year, month, 1)
    
    # 月の最後の日を計算（12月の場合は特別処理）
    if month == 12:
        last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = datetime(year, month + 1, 1) - timedelta(days=1)
    
    # 祝日情報を取得
    holidays = get_japanese_holidays(year)
    
    # Prophetモデルで月間予測を実行
    try:
        # 予測日付の生成
        future_dates = pd.DataFrame({
            'ds': pd.date_range(start=first_day, end=last_day, freq='D')
        })
        
        # Prophetモデルで予測を実行
        forecast = prediction_model.prophet_model.predict(future_dates)
        
        # 予測結果を整形
        monthly_predictions = {}
        for i, row in forecast.iterrows():
            date_str = row['ds'].strftime('%Y-%m-%d')
            prediction = float(row['yhat'])
            evaluation = prediction_model.evaluate_inpatient_level(prediction)
            
            # 祝日情報を追加
            is_holiday = date_str in holidays
            holiday_name = holidays.get(date_str, '')
            
            monthly_predictions[date_str] = {
                'level': evaluation['level'],
                'color': evaluation['color'],
                'label': evaluation['label'],
                'prediction': round(prediction, 1),
                'is_holiday': is_holiday,
                'holiday_name': holiday_name
            }
        
        return render_template('monthly_calendar.html',
                             current_year=year,
                             current_month=month,
                             monthly_predictions=monthly_predictions)
                             
    except Exception as e:
        print(f"Error in monthly calendar prediction: {str(e)}")
        return render_template('monthly_calendar.html',
                             current_year=year,
                             current_month=month,
                             monthly_predictions={},
                             error_message=str(e))

@main.route('/data_analysis')
def data_analysis():
    return render_template('data_analysis.html')

@main.route('/scenario_comparison')
def scenario_comparison():
    return render_template('scenario_comparison.html')

@main.route('/predict', methods=['POST'])
def predict():
    try:
        # リクエストの詳細をデバッグ出力
        print("\n=== 予測リクエストの詳細 ===")
        print(f"Method: {request.method}")
        print(f"Headers: {dict(request.headers)}")
        print(f"Content-Type: {request.content_type}")
        
        if request.is_json:
            print("\nJSONデータを処理します")
            try:
                data = request.get_json(force=True)
                print(f"受信したJSONデータ: {data}")
            except Exception as e:
                print(f"JSONデータの解析に失敗: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': 'JSONデータの解析に失敗しました',
                    'error_type': 'json_parse_error',
                    'error_details': str(e)
                }), 400
        else:
            print("\nフォームデータを処理します")
            data = request.form.to_dict()
            print(f"受信したフォームデータ: {data}")
        
        required_fields = ['date', 'total_outpatient', 'intro_outpatient', 'er_patients', 'bed_count', 'y']
        
        # 必須フィールドの存在チェック
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            error_msg = f'必須フィールドが不足しています: {", ".join(missing_fields)}'
            print(f"\nエラー: {error_msg}")
            print(f"受信したデータ: {data}")
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_type': 'missing_fields',
                'missing_fields': missing_fields,
                'received_data': data
            }), 400
        
        # データの型変換とバリデーション
        try:
            prediction_date = pd.to_datetime(data['date'])
            input_date = prediction_date - timedelta(days=1)  # 前日の日付を計算
            
            # 予測用のデータを準備
            prediction_data = {
                'date': str(data['date']),
                'total_outpatient': int(float(data['total_outpatient'])),
                'intro_outpatient': int(float(data['intro_outpatient'])),
                'er_patients': int(float(data['er_patients'])),
                'bed_count': int(float(data['bed_count']))
            }
            
            # Supabaseに保存するデータを準備（前日のデータとして保存）
            input_data = {
                'date': input_date.strftime('%Y-%m-%d'),  # 前日の日付
                'total_outpatient': int(float(data['total_outpatient'])),  # 外来患者数
                'intro_outpatient': int(float(data['intro_outpatient'])),  # 紹介患者数
                'er_patients': int(float(data['er_patients'])),  # 救急搬送患者数
                'bed_count': int(float(data['bed_count'])),  # 入院患者数
                'y': int(float(data['y']))  # 新入院患者数
            }
            
            print(f"\n変換後のデータ: {prediction_data}")
            print(f"Supabaseに保存するデータ（前日）: {input_data}")
            
            # 値の範囲チェック
            validations = [
                (0 <= prediction_data['total_outpatient'] <= 2000, "外来患者数は0から2000の間である必要があります"),
                (0 <= prediction_data['intro_outpatient'] <= 200, "紹介患者数は0から200の間である必要があります"),
                (0 <= prediction_data['er_patients'] <= 100, "救急搬送患者数は0から100の間である必要があります"),
                (0 <= prediction_data['bed_count'] <= 1000, "入院患者数は0から1000の間である必要があります"),
                (0 <= int(float(data['y'])) <= 100, "新入院患者数は0から100の間である必要があります")
            ]
            
            for is_valid, error_message in validations:
                if not is_valid:
                    print(f"\nバリデーションエラー: {error_message}")
                    return jsonify({
                        'status': 'error',
                        'message': error_message,
                        'error_type': 'validation_error',
                        'received_data': data
                    }), 400
            
            # Supabaseにデータを保存
            try:
                save_result = save_input_data(input_data)
                print(f"\nデータを保存しました: {save_result}")
            except Exception as e:
                print(f"\nデータ保存中にエラーが発生: {str(e)}")
                import traceback
                print(f"スタックトレース:\n{traceback.format_exc()}")
                # データ保存エラーはログに記録するが、予測は続行する
            
            # 予測を実行
            print("\n予測を実行します")
            result = prediction_model.predict(**prediction_data)
            print(f"予測結果: {result}")
            return jsonify(result)
            
        except (ValueError, TypeError) as e:
            error_msg = f'データの型変換に失敗しました: {str(e)}'
            print(f"\nエラー: {error_msg}")
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_type': 'type_conversion_error',
                'error_details': str(e),
                'received_data': data
            }), 400
            
    except Exception as e:
        import traceback
        error_msg = f'予期せぬエラーが発生しました: {str(e)}'
        print(f"\nエラー: {error_msg}")
        print(f"スタックトレース:\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error_type': 'unexpected_error',
            'error_details': str(e),
            'stack_trace': traceback.format_exc()
        }), 500

@main.route('/predict/weekly', methods=['POST'])
def predict_weekly():
    """1週間分の予測を実行する（Prophetモデル使用）"""
    try:
        print("\n=== 週間予測リクエストの詳細 ===")
        data = request.get_json()
        print(f"受信したデータ: {data}")
        
        required_fields = ['date', 'bed_count']
        
        # 必須フィールドのチェック
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            error_msg = f'必須フィールドが不足しています: {", ".join(missing_fields)}'
            print(f"\nエラー: {error_msg}")
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_type': 'missing_fields',
                'missing_fields': missing_fields
            }), 400

        try:
            start_date = pd.to_datetime(data['date'])
            bed_count = int(data['bed_count'])
            
            if not (0 <= bed_count <= 1000):
                return jsonify({
                    'status': 'error',
                    'message': '入院患者数は0から1000の間である必要があります',
                    'error_type': 'validation_error'
                }), 400
                
            print(f"\n予測期間: {start_date} から 7日間")
            print(f"入院患者数: {bed_count}")
            
            # 7日分の日付を生成
            future_dates = pd.DataFrame({
                'ds': [start_date + timedelta(days=i) for i in range(7)]
            })
            
            print("\nProphetモデルで予測を実行")
            forecast = prediction_model.prophet_model.predict(future_dates)
            print(f"予測完了: {len(forecast)}件")
            
            # 予測結果を整形
            predictions = []
            dates = []
            
            for _, row in forecast.iterrows():
                date_str = row['ds'].strftime('%Y-%m-%d')
                prediction = float(row['yhat'])
                
                # 予測値の評価
                evaluation = prediction_model.evaluate_inpatient_level(prediction)
                
                predictions.append({
                    'date': date_str,
                    'value': round(prediction, 1),
                    'level': evaluation['level'],
                    'color': evaluation['color'],
                    'label': evaluation['label']
                })
                dates.append(date_str)
            
            print(f"予測結果: {predictions}")
            
            return jsonify({
                'status': 'success',
                'predictions': predictions,
                'dates': dates
            })
            
        except ValueError as e:
            error_msg = f'データの型変換に失敗しました: {str(e)}'
            print(f"\nエラー: {error_msg}")
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_type': 'type_conversion_error',
                'error_details': str(e)
            }), 400
            
    except Exception as e:
        import traceback
        error_msg = f'週間予測中にエラーが発生しました: {str(e)}'
        print(f"\nエラー: {error_msg}")
        print(f"スタックトレース:\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error_type': 'prediction_error',
            'error_details': str(e),
            'stack_trace': traceback.format_exc()
        }), 500

@main.route('/update_actual', methods=['POST'])
def update_actual():
    """実際の入院患者数を更新"""
    try:
        data = request.get_json()
        date = data.get('date')
        actual_value = data.get('actual_value')

        if not date or actual_value is None:
            return jsonify({
                'status': 'error',
                'message': '日付と実際の値が必要です'
            })

        # 実際の値を更新
        if prediction_model.prediction_manager.update_actual_value(date, actual_value):
            # メトリクスを計算
            metrics = prediction_model.prediction_manager.calculate_metrics()
            return jsonify({
                'status': 'success',
                'message': '実際の値を更新しました',
                'metrics': metrics
            })
        else:
            return jsonify({
                'status': 'error',
                'message': '指定された日付の予測が見つかりません'
            })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@main.route('/predictions', methods=['GET'])
def get_predictions():
    """全ての予測データを取得"""
    try:
        predictions = prediction_model.prediction_manager.get_all_predictions()
        return jsonify({
            'status': 'success',
            'predictions': predictions.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@main.route('/get_monthly_predictions', methods=['POST'])
def get_monthly_predictions():
    try:
        data = request.get_json()
        year = int(data.get('year', datetime.now().year))
        month = int(data.get('month', datetime.now().month))
        
        # 祝日データを取得
        holidays = get_japanese_holidays(year)
        
        # 月の最初の日と最後の日を取得
        first_day = datetime(year, month, 1)
        if month == 12:
            last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = datetime(year, month + 1, 1) - timedelta(days=1)
        
        # カレンダーの日付を生成
        calendar_days = []
        
        # 月の最初の日の曜日を取得（0: 日曜日, 6: 土曜日）
        first_weekday = first_day.strftime("%w")
        first_weekday = int(first_weekday)
        
        # 前月の日付を追加
        if first_weekday > 0:
            prev_month = first_day - relativedelta(months=1)
            prev_month_days = calendar.monthrange(prev_month.year, prev_month.month)[1]
            for i in range(first_weekday):
                prev_date = prev_month.replace(day=prev_month_days - first_weekday + i + 1)
                calendar_days.append({
                    'date': prev_date.day,
                    'value': None,
                    'level': 'prev-month',
                    'is_holiday': False,
                    'weekday': int(prev_date.strftime("%w"))
                })
        
        # 当月の日付を追加
        current_date = first_day
        predictions_list = []
        
        while current_date <= last_day:
            # 祝日かどうかをチェック
            is_holiday = current_date.strftime('%Y-%m-%d') in holidays
            
            # Prophetモデルで予測を実行
            future_dates = pd.DataFrame({
                'ds': [current_date]
            })
            forecast = prediction_model.prophet_model.predict(future_dates)
            prediction = float(forecast['yhat'].iloc[0])
            
            predictions_list.append(prediction)
            calendar_days.append({
                'date': current_date.day,
                'value': round(prediction, 1),
                'level': 'pending',
                'is_holiday': is_holiday,
                'weekday': int(current_date.strftime("%w"))
            })
            
            current_date += timedelta(days=1)
        
        # 翌月の日付を追加（6週間分のカレンダーにする）
        last_weekday = int(last_day.strftime("%w"))
        remaining_days = 6 - last_weekday
        if remaining_days > 0:
            next_month = last_day + timedelta(days=1)
            for i in range(remaining_days):
                next_date = next_month + timedelta(days=i)
                calendar_days.append({
                    'date': next_date.day,
                    'value': None,
                    'level': 'next-month',
                    'is_holiday': False,
                    'weekday': int(next_date.strftime("%w"))
                })
        
        # 予測値の統計を計算
        if predictions_list:
            mean_pred = np.mean(predictions_list)
            std_pred = np.std(predictions_list)
            
            # 混雑度レベルの閾値を設定
            high_threshold = mean_pred + std_pred
            medium_high_threshold = mean_pred + (std_pred * 0.5)
            medium_low_threshold = mean_pred - (std_pred * 0.5)
            
            # レベルを設定
            for day in calendar_days:
                if day['value'] is not None:
                    if day['value'] >= high_threshold:
                        day['level'] = 'high'
                    elif day['value'] >= medium_high_threshold:
                        day['level'] = 'medium-high'
                    elif day['value'] >= medium_low_threshold:
                        day['level'] = 'medium-low'
                    else:
                        day['level'] = 'low'
        
        return jsonify({
            'status': 'success',
            'predictions': calendar_days,
            'thresholds': {
                'high': round(high_threshold, 1),
                'medium_high': round(medium_high_threshold, 1),
                'medium_low': round(medium_low_threshold, 1),
                'mean': round(mean_pred, 1)
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@main.route('/get_seasonal_trends')
def get_seasonal_trends():
    try:
        # 1年分の予測データを生成
        start_date = datetime.now()
        predictions = []
        dates = []
        
        for i in range(365):
            current_date = start_date + timedelta(days=i)
            # デフォルト値を取得
            defaults = get_default_values(current_date)
            
            # 予測モデルを使用して予測を実行
            result = prediction_model.predict(
                date=current_date.strftime('%Y-%m-%d'),
                total_outpatient=defaults['total_outpatient'],
                intro_outpatient=defaults['intro_outpatient'],
                er_patients=defaults['er_patients'],
                bed_count=defaults['bed_count']
            )
            
            if result['status'] == 'success':
                predictions.append(result['prediction'])
                dates.append(current_date)
        
        # 月別の平均を計算
        df = pd.DataFrame({
            'date': dates,
            'prediction': predictions
        })
        df['month'] = df['date'].dt.month
        monthly_avg = df.groupby('month')['prediction'].mean()
        
        # プロット用のデータを準備
        data = [{
            'x': list(range(1, 13)),
            'y': monthly_avg.values.tolist(),
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': '月別平均'
        }]
        
        return jsonify({
            'status': 'success',
            'data': data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@main.route('/get_weekday_analysis')
def get_weekday_analysis():
    try:
        # 1週間分の予測データを生成
        start_date = datetime.now()
        predictions = []
        weekdays = []
        
        for i in range(7):
            current_date = start_date + timedelta(days=i)
            # デフォルト値を取得
            defaults = get_default_values(current_date)
            
            # 予測モデルを使用して予測を実行
            result = prediction_model.predict(
                date=current_date.strftime('%Y-%m-%d'),
                total_outpatient=defaults['total_outpatient'],
                intro_outpatient=defaults['intro_outpatient'],
                er_patients=defaults['er_patients'],
                bed_count=defaults['bed_count']
            )
            
            if result['status'] == 'success':
                predictions.append(result['prediction'])
                weekdays.append(current_date.strftime('%A'))
        
        # プロット用のデータを準備
        data = [{
            'x': weekdays,
            'y': predictions,
            'type': 'bar',
            'name': '曜日別予測'
        }]
        
        return jsonify({
            'status': 'success',
            'data': data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@main.route('/get_monthly_analysis')
def get_monthly_analysis():
    try:
        # 1年分の予測データを生成
        start_date = datetime.now()
        predictions = []
        months = []
        
        for i in range(365):
            current_date = start_date + timedelta(days=i)
            # デフォルト値を取得
            defaults = get_default_values(current_date)
            
            # 予測モデルを使用して予測を実行
            result = prediction_model.predict(
                date=current_date.strftime('%Y-%m-%d'),
                total_outpatient=defaults['total_outpatient'],
                intro_outpatient=defaults['intro_outpatient'],
                er_patients=defaults['er_patients'],
                bed_count=defaults['bed_count']
            )
            
            if result['status'] == 'success':
                predictions.append(result['prediction'])
                months.append(current_date.strftime('%B'))
        
        # 月別の平均を計算
        df = pd.DataFrame({
            'month': months,
            'prediction': predictions
        })
        monthly_avg = df.groupby('month')['prediction'].mean()
        
        # プロット用のデータを準備
        data = [{
            'x': monthly_avg.index.tolist(),
            'y': monthly_avg.values.tolist(),
            'type': 'bar',
            'name': '月別平均'
        }]
        
        return jsonify({
            'status': 'success',
            'data': data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@main.route('/compare_scenarios', methods=['POST'])
def compare_scenarios():
    try:
        data = request.get_json()
        base_date = datetime.strptime(data['base_date'], '%Y-%m-%d')
        scenarios = data['scenarios']
        
        # 各シナリオの予測を実行
        results = []
        for i, scenario in enumerate(scenarios):
            result = prediction_model.predict(
                date=base_date.strftime('%Y-%m-%d'),
                total_outpatient=scenario['total_outpatient'],
                intro_outpatient=scenario['intro_outpatient'],
                er_patients=scenario['er_patients'],
                bed_count=scenario['bed_count']
            )
            
            if result['status'] == 'success':
                results.append({
                    'scenario': i + 1,
                    'prediction': result['prediction']
                })
        
        # プロット用のデータを準備
        plot_data = [{
            'x': [f'シナリオ {r["scenario"]}' for r in results],
            'y': [r['prediction'] for r in results],
            'type': 'bar',
            'name': '予測結果'
        }]
        
        # テーブル用のHTMLを生成
        table_html = '''
        <table class="table">
            <thead>
                <tr>
                    <th>シナリオ</th>
                    <th>予測入院患者数</th>
                </tr>
            </thead>
            <tbody>
        '''
        for r in results:
            table_html += f'''
                <tr>
                    <td>シナリオ {r["scenario"]}</td>
                    <td>{r["prediction"]:.1f}</td>
                </tr>
            '''
        table_html += '''
            </tbody>
        </table>
        '''
        
        return jsonify({
            'status': 'success',
            'data': plot_data,
            'table_html': table_html
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }) 