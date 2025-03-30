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

main = Blueprint('main', __name__)

# 予測モデルのインスタンスを作成
prediction_model = PredictionModel()

def get_japanese_holidays(year):
    """日本の祝日を取得"""
    try:
        response = requests.get(f'https://holidays-jp.github.io/api/v1/date.json')
        if response.status_code == 200:
            holidays = response.json()
            return {k: v for k, v in holidays.items() if k.startswith(str(year))}
    except:
        # エラーの場合は空の辞書を返す
        pass
    return {}

def get_default_values(date):
    """日付に基づいてデフォルト値を設定"""
    date_obj = pd.to_datetime(date)
    month = date_obj.month
    
    # 季節による調整のみを残す
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
    
    # 月間予測を生成
    monthly_predictions = {}
    current_date = first_day
    while current_date <= last_day:
        date_str = current_date.strftime('%Y-%m-%d')
        # デフォルト値を使用して予測を実行
        default_values = get_default_values(date_str)
        prediction = prediction_model.predict(
            date=date_str,
            total_outpatient=default_values['total_outpatient'],
            intro_outpatient=default_values['intro_outpatient'],
            er_patients=default_values['er_patients'],
            bed_count=default_values['bed_count']
        )
        monthly_predictions[date_str] = {
            'level': prediction['evaluation']['level'],
            'color': prediction['evaluation']['color'],
            'label': prediction['evaluation']['label']
        }
        current_date += timedelta(days=1)
    
    return render_template('monthly_calendar.html',
                         current_year=year,
                         current_month=month,
                         monthly_predictions=monthly_predictions)

@main.route('/data_analysis')
def data_analysis():
    return render_template('data_analysis.html')

@main.route('/scenario_comparison')
def scenario_comparison():
    return render_template('scenario_comparison.html')

@main.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            
            # 入力データの取得
            date = data.get('prediction_date')
            total_outpatient = data.get('total_outpatient', 600)
            intro_outpatient = data.get('intro_outpatient', 30)
            er_patients = data.get('er_patients', 20)
            bed_count = data.get('bed_count', 300)
            
            # RandomForestモデルで当日予測を実行
            result = prediction_model.predict_daily(
                date=date,
                total_outpatient=total_outpatient,
                intro_outpatient=intro_outpatient,
                er_patients=er_patients,
                bed_count=bed_count
            )

            if result['status'] != 'success':
                return jsonify(result)

            return jsonify({
                'status': 'success',
                'prediction': result['daily_prediction'],
                'evaluation': result['evaluation']
            })

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'予測中にエラーが発生しました: {str(e)}'
            })

    return jsonify({
        'status': 'error',
        'message': '不正なリクエストです'
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
            
            # Prophetモデルで週間予測を実行
            result = prediction_model.predict_weekly(
                date=current_date.strftime('%Y-%m-%d'),
                bed_count=300
            )
            
            if result['status'] == 'success':
                prediction = result['daily_prediction']
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
                    'weekday': int(next_date.strftime("%w"))  # 0: 日曜日, 6: 土曜日
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