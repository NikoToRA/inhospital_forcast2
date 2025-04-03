from datetime import datetime, timedelta
from supabase import create_client, Client
from config import Config
import traceback
import jpholiday  # 日本の祝日判定用
import os

# Supabaseクライアントの初期化
supabase: Client = create_client(
    Config.SUPABASE_URL,
    Config.SUPABASE_KEY
)

def init_database():
    """データベースの初期化"""
    try:
        # テーブルの存在確認
        response = supabase.table('training_data').select('id').limit(1).execute()
        print("training_dataテーブルが存在します")
    except Exception as e:
        print("training_dataテーブルが存在しません。作成します...")
        # テーブルの作成
        with open('app/schema.sql', 'r') as f:
            sql = f.read()
        supabase.rpc('exec_sql', {'sql': sql}).execute()
        print("テーブルを作成しました")

def add_training_data(date, mon, tue, wed, thu, fri, sat, sun, public_holiday, public_holiday_previous_day, total_outpatient, intro_outpatient, er, bed_count, y):
    """トレーニングデータの追加"""
    try:
        data = {
            'date': date,
            'mon': mon,
            'tue': tue,
            'wed': wed,
            'thu': thu,
            'fri': fri,
            'sat': sat,
            'sun': sun,
            'public_holiday': public_holiday,
            'public_holiday_previous_day': public_holiday_previous_day,
            'total_outpatient': total_outpatient,
            'intro_outpatient': intro_outpatient,
            'er': er,
            'bed_count': bed_count,
            'y': y
        }
        response = supabase.table('training_data').insert(data).execute()
        return response.data
    except Exception as e:
        print(f"Error adding training data: {e}")
        return None

def get_training_data():
    """トレーニングデータを取得"""
    response = supabase.table('training_data').select('*').execute()
    return response.data

def add_model_metrics(metrics):
    """モデルのメトリクスを保存"""
    response = supabase.table('model_metrics').insert(metrics).execute()
    return response.data

def get_model_metrics():
    """モデルの評価指標の取得"""
    try:
        response = supabase.table('model_metrics').select("*").order('date').execute()
        return response.data
    except Exception as e:
        print(f"Error getting model metrics: {e}")
        return []

def save_prediction(date, predicted_value, actual_value=None):
    """予測結果の保存"""
    try:
        data = {
            'date': date,
            'predicted_value': predicted_value,
            'actual_value': actual_value
        }
        response = supabase.table('predictions').insert(data).execute()
        return response.data
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return None

def get_predictions():
    """予測結果の取得"""
    try:
        response = supabase.table('predictions').select("*").order('date').execute()
        return response.data
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return []

def get_prediction_for_date(date):
    """特定の日付の予測を取得"""
    try:
        response = supabase.table('predictions').select("*").eq('date', date).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"Error getting prediction for date: {e}")
        return None

def get_weekday_info(date_str):
    """日付から曜日情報を生成"""
    date = datetime.strptime(date_str, '%Y-%m-%d')
    return {
        'mon': date.weekday() == 0,
        'tue': date.weekday() == 1,
        'wed': date.weekday() == 2,
        'thu': date.weekday() == 3,
        'fri': date.weekday() == 4,
        'sat': date.weekday() == 5,
        'sun': date.weekday() == 6
    }

def get_holiday_info(date_str):
    """日付から祝日情報を生成"""
    date = datetime.strptime(date_str, '%Y-%m-%d')
    prev_date = date - timedelta(days=1)
    
    return {
        'public_holiday': jpholiday.is_holiday(date),
        'public_holiday_previous_day': jpholiday.is_holiday(prev_date)
    }

def save_input_data(data):
    """入力データをSupabaseに保存"""
    try:
        # 日付を文字列に変換（既に文字列の場合はそのまま使用）
        formatted_data = data.copy()
        if isinstance(formatted_data['date'], str):
            date_str = formatted_data['date']
        else:
            date_str = formatted_data['date'].strftime('%Y-%m-%d')
        
        # カラム名の修正
        if 'er_patients' in formatted_data:
            formatted_data['er'] = formatted_data.pop('er_patients')
        
        # 日付を設定
        formatted_data['date'] = date_str
        
        # 曜日情報を追加
        weekday_info = get_weekday_info(date_str)
        formatted_data.update(weekday_info)
        
        # 祝日情報を追加
        holiday_info = get_holiday_info(date_str)
        formatted_data.update(holiday_info)
        
        print(f"\n保存するデータ: {formatted_data}")
        
        try:
            # まず、同じ日付のデータが存在するか確認
            existing_data = supabase.table('training_data').select("*").eq('date', formatted_data['date']).execute()
            print(f"\n既存データの確認結果: {existing_data.data}")
            
            if existing_data.data:
                print(f"警告: {formatted_data['date']}の日付のデータが既に存在します")
                # 既存のデータを更新
                response = supabase.table('training_data').update(formatted_data).eq('date', formatted_data['date']).execute()
            else:
                # 新規データを挿入
                response = supabase.table('training_data').insert(formatted_data).execute()
            
            if not response:
                print("\nSupabase response is None")
                raise Exception("データの保存に失敗しました: レスポンスがNoneです")
            
            print(f"\nSupabase response details:")
            print(f"Response object: {response}")
            if hasattr(response, 'data'):
                print(f"Response data: {response.data}")
            if hasattr(response, 'error'):
                print(f"Response error: {response.error}")
            if hasattr(response, 'status_code'):
                print(f"Status code: {response.status_code}")
            
            if not response.data:
                raise Exception("データの保存に失敗しました: レスポンスデータが空です")
            
            print(f"\nデータを保存しました:")
            print(f"Response data: {response.data}")
            return response.data
            
        except Exception as e:
            print("\nSupabase insertion error details:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Error attributes: {dir(e)}")  # エラーオブジェクトの全属性を表示
            if hasattr(e, 'response'):
                print(f"Error response: {e.response}")
            if hasattr(e, 'status_code'):
                print(f"Error status code: {e.status_code}")
            print(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"Supabaseへのデータ挿入に失敗: {str(e)}")
            
    except Exception as e:
        error_msg = f"データ保存中にエラーが発生: {str(e)}"
        print(f"\n{error_msg}")
        print(f"Error type: {type(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise Exception(error_msg)

# データベースの初期化
init_database() 