import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

def check_model_exists():
    """モデルファイルの存在を確認する"""
    if not os.path.exists('fixed_rf_model.joblib'):
        print("エラー: fixed_rf_model.joblib が見つかりません。")
        print("モデルファイルが正しいディレクトリに配置されているか確認してください。")
        return False
    return True

def create_test_data(date):
    """テストデータを生成する"""
    try:
        # 日付から各種情報を抽出
        weekday = date.weekday()
        month = date.month
        day = date.day
        
        # 曜日の1-hotエンコーディング
        test_data = pd.DataFrame({
            'mon': [1 if weekday == 0 else 0],
            'tue': [1 if weekday == 1 else 0],
            'wed': [1 if weekday == 2 else 0],
            'thu': [1 if weekday == 3 else 0],
            'fri': [1 if weekday == 4 else 0],
            'sat': [1 if weekday == 5 else 0],
            'sun': [1 if weekday == 6 else 0],
            'public_holiday': [0],
            'public_holiday_previous_day': [0],
            'total_outpatient': [600],
            'intro_outpatient': [30],
            'ER': [20],
            'bed_count': [300]
        })
        
        # 季節による外来患者数の調整
        if month in [12, 1, 2]:  # 冬季
            test_data['total_outpatient'] *= 1.2
            test_data['ER'] *= 1.3
        elif month in [6, 7, 8]:  # 夏季
            test_data['total_outpatient'] *= 0.9
            test_data['ER'] *= 1.1
        
        # 月初めと月末の調整
        if day <= 5:  # 月初め
            test_data['total_outpatient'] *= 1.1
        elif day >= 25:  # 月末
            test_data['total_outpatient'] *= 0.9
        
        return test_data
    except Exception as e:
        print(f"テストデータ生成中にエラーが発生しました: {e}")
        return None

def predict_single_date(date):
    """特定の日付の予測を実行する"""
    try:
        if not check_model_exists():
            return None
        
        model = joblib.load('fixed_rf_model.joblib')
        test_data = create_test_data(date)
        
        if test_data is None:
            return None
        
        prediction = model.predict(test_data)
        return prediction[0]
    except Exception as e:
        print(f"予測実行中にエラーが発生しました: {e}")
        return None

def analyze_seasonal_trends():
    """季節トレンドの分析を実行する"""
    try:
        if not check_model_exists():
            return
        
        # モデルを読み込む
        model = joblib.load('fixed_rf_model.joblib')
        print("モデルを正常に読み込みました")
        
        # 1年分の予測を実行
        start_date = datetime.today()
        predictions = []
        dates = []
        
        for i in range(365):  # 1年分
            current_date = start_date + timedelta(days=i)
            test_data = create_test_data(current_date)
            
            if test_data is not None:
                prediction = model.predict(test_data)
                predictions.append(prediction[0])
                dates.append(current_date)
        
        if not predictions:
            print("予測データが生成できませんでした。")
            return
        
        # 結果をデータフレームにまとめる
        results = pd.DataFrame({
            'date': dates,
            'predicted_admissions': predictions,
            'month': [d.month for d in dates],
            'day': [d.day for d in dates],
            'weekday': [d.weekday() for d in dates]
        })
        
        # 月ごとの統計
        print("\n=== 月ごとの統計 ===")
        monthly_stats = results.groupby('month')['predicted_admissions'].agg(['mean', 'std', 'min', 'max'])
        for month, stats in monthly_stats.iterrows():
            print(f"\n{month}月:")
            print(f"平均: {stats['mean']:.1f}人")
            print(f"標準偏差: {stats['std']:.2f}")
            print(f"最小: {stats['min']:.1f}人")
            print(f"最大: {stats['max']:.1f}人")
        
        # 月の週ごとの統計
        print("\n=== 月の週ごとの統計 ===")
        results['week_of_month'] = (results['day'] - 1) // 7 + 1
        weekly_stats = results.groupby('week_of_month')['predicted_admissions'].agg(['mean', 'std'])
        for week, stats in weekly_stats.iterrows():
            print(f"\n第{week}週:")
            print(f"平均: {stats['mean']:.1f}人")
            print(f"標準偏差: {stats['std']:.2f}")
        
        # 季節ごとの統計
        print("\n=== 季節ごとの統計 ===")
        results['season'] = results['month'].apply(lambda x: 
            '冬' if x in [12, 1, 2] else
            '春' if x in [3, 4, 5] else
            '夏' if x in [6, 7, 8] else '秋')
        
        seasonal_stats = results.groupby('season')['predicted_admissions'].agg(['mean', 'std', 'min', 'max'])
        for season, stats in seasonal_stats.iterrows():
            print(f"\n{season}:")
            print(f"平均: {stats['mean']:.1f}人")
            print(f"標準偏差: {stats['std']:.2f}")
            print(f"最小: {stats['min']:.1f}人")
            print(f"最大: {stats['max']:.1f}人")
        
    except Exception as e:
        print(f"分析中にエラーが発生しました: {e}")

def main():
    """メイン関数"""
    print("入院予測分析システム")
    print("1. 特定の日付の予測")
    print("2. 季節トレンドの分析")
    print("3. 終了")
    
    while True:
        try:
            choice = input("\n実行する機能を選択してください (1-3): ")
            
            if choice == "1":
                try:
                    date_str = input("予測したい日付を入力してください (YYYY-MM-DD): ")
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                    prediction = predict_single_date(date)
                    if prediction is not None:
                        print(f"\n{date_str}の予測入院者数: {prediction:.1f}人")
                except ValueError:
                    print("正しい日付形式で入力してください (YYYY-MM-DD)")
            
            elif choice == "2":
                analyze_seasonal_trends()
            
            elif choice == "3":
                print("プログラムを終了します。")
                break
            
            else:
                print("1-3の数字を入力してください。")
        
        except KeyboardInterrupt:
            print("\nプログラムを終了します。")
            break
        except Exception as e:
            print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main() 