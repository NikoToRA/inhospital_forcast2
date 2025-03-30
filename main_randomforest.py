import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os

# モデルを修正して保存する関数
def fix_model_compatibility():
    try:
        # 元のモデルを読み込む
        print("元のモデルを読み込み中...")
        model = joblib.load('fixed_rf_model.joblib')
        print("モデルを正常に読み込みました")
        
        # モデルの内部構造を修正
        # monotonic_cst属性が欠落しているため、追加する
        for estimator in model.estimators_:
            if not hasattr(estimator, 'monotonic_cst'):
                estimator.monotonic_cst = None
        
        # 修正したモデルを保存
        joblib.dump(model, 'fixed_rf_model.joblib')
        print("修正したモデルを 'fixed_rf_model.joblib' として保存しました")
        return True
    except Exception as e:
        print(f"モデルの修正に失敗しました: {e}")
        return False

# CSVデータを参考にした現実的なテストデータを作成
def create_realistic_test_data():
    # 曜日を1-hotエンコーディングで表現（月曜日を例として）
    mon, tue, wed, thu, fri, sat, sun = 1, 0, 0, 0, 0, 0, 0
    
    # 祝日フラグ（通常の平日を想定）
    public_holiday = 0
    public_holiday_previous_day = 0
    
    # 病院の状況を表す特徴量（CSVデータの平均的な値を使用）
    total_outpatient = 500 # 総外来患者数
    intro_outpatient = 20   # 紹介外来患者数
    er = 15                 # 救急患者数
    bed_count = 280         # ベッド数
    
    # テストデータを作成
    test_data = pd.DataFrame({
        'mon': [mon],
        'tue': [tue],
        'wed': [wed],
        'thu': [thu],
        'fri': [fri],
        'sat': [sat],
        'sun': [sun],
        'public_holiday': [public_holiday],
        'public_holiday_previous_day': [public_holiday_previous_day],
        'total_outpatient': [total_outpatient],
        'intro_outpatient': [intro_outpatient],
        'ER': [er],
        'bed_count': [bed_count]
    })
    
    return test_data

# CSVデータから実際のシナリオを作成
def create_scenarios_from_csv():
    try:
        # CSVファイルを読み込む
        df = pd.read_csv('ultimate_pickup_data.csv')
        
        # NaN値を含む行を削除
        df = df.dropna()
        
        # 特徴量と目標変数を分離
        X = df.drop(['date', 'y'], axis=1)
        y = df['y']
        
        # 代表的なシナリオを選択
        # 1. 月曜日で外来患者数が多い日
        monday_high = df[(df['mon'] == 1) & (df['total_outpatient'] > 700)].iloc[0]
        
        # 2. 火曜日で通常の外来患者数
        tuesday_normal = df[(df['tue'] == 1) & (df['total_outpatient'] > 500) & (df['total_outpatient'] < 700)].iloc[0]
        
        # 3. 水曜日で外来患者数が多い日
        wednesday_high = df[(df['wed'] == 1) & (df['total_outpatient'] > 700)].iloc[0]
        
        # 4. 土曜日で外来患者数が少ない日
        saturday_low = df[(df['sat'] == 1) & (df['total_outpatient'] < 250)].iloc[0]
        
        # 5. 祝日
        holiday = df[df['public_holiday'] == 1].iloc[0]
        
        scenarios = [
            monday_high,
            tuesday_normal,
            wednesday_high,
            saturday_low,
            holiday
        ]
        
        return scenarios, df
    
    except Exception as e:
        print(f"CSVファイルの読み込みに失敗しました: {e}")
        return None, None

# 複数のシナリオを作成して予測
def predict_multiple_scenarios(model):
    # CSVデータからシナリオを作成
    scenarios, df = create_scenarios_from_csv()
    
    if scenarios is None:
        # CSVデータが読み込めない場合は手動でシナリオを作成
        scenarios = [
            # 月曜日、外来患者数が多い
            {'mon': 1, 'tue': 0, 'wed': 0, 'thu': 0, 'fri': 0, 'sat': 0, 'sun': 0, 
             'public_holiday': 0, 'public_holiday_previous_day': 0, 
             'total_outpatient': 800, 'intro_outpatient': 30, 'ER': 20, 'bed_count': 280,
             'description': '月曜日、外来患者数が多い'},
            
            # 火曜日、通常の外来患者数
            {'mon': 0, 'tue': 1, 'wed': 0, 'thu': 0, 'fri': 0, 'sat': 0, 'sun': 0, 
             'public_holiday': 0, 'public_holiday_previous_day': 0, 
             'total_outpatient': 600, 'intro_outpatient': 20, 'ER': 15, 'bed_count': 280,
             'description': '火曜日、通常の外来患者数'},
            
            # 水曜日、外来患者数が多い
            {'mon': 0, 'tue': 0, 'wed': 1, 'thu': 0, 'fri': 0, 'sat': 0, 'sun': 0, 
             'public_holiday': 0, 'public_holiday_previous_day': 0, 
             'total_outpatient': 750, 'intro_outpatient': 25, 'ER': 15, 'bed_count': 280,
             'description': '水曜日、外来患者数が多い'},
            
            # 土曜日、少ない外来患者数
            {'mon': 0, 'tue': 0, 'wed': 0, 'thu': 0, 'fri': 0, 'sat': 1, 'sun': 0, 
             'public_holiday': 0, 'public_holiday_previous_day': 0, 
             'total_outpatient': 200, 'intro_outpatient': 5, 'ER': 15, 'bed_count': 280,
             'description': '土曜日、少ない外来患者数'},
            
            # 祝日
            {'mon': 0, 'tue': 1, 'wed': 0, 'thu': 0, 'fri': 0, 'sat': 0, 'sun': 0, 
             'public_holiday': 1, 'public_holiday_previous_day': 0, 
             'total_outpatient': 100, 'intro_outpatient': 3, 'ER': 15, 'bed_count': 280,
             'description': '祝日（火曜日）'}
        ]
        
        results = []
        
        for scenario in scenarios:
            description = scenario.pop('description')
            df = pd.DataFrame([scenario])
            
            try:
                prediction = model.predict(df)
                results.append((description, prediction[0]))
            except Exception as e:
                print(f"予測に失敗しました（{description}）: {e}")
                results.append((description, "予測失敗"))
    else:
        # CSVデータからシナリオを作成した場合
        results = []
        descriptions = [
            "月曜日、外来患者数が多い",
            "火曜日、通常の外来患者数",
            "水曜日、外来患者数が多い",
            "土曜日、少ない外来患者数",
            "祝日"
        ]
        
        for i, scenario in enumerate(scenarios):
            # 日付と目標変数を除外
            scenario_data = scenario.drop(['date', 'y'])
            
            # DataFrameに変換
            df = pd.DataFrame([scenario_data])
            
            try:
                prediction = model.predict(df)
                actual = scenario['y']
                results.append((descriptions[i], prediction[0], actual))
            except Exception as e:
                print(f"予測に失敗しました（{descriptions[i]}）: {e}")
                results.append((descriptions[i], "予測失敗", None))
    
    return results

# メイン処理
def main():
    print("=== 入院患者数予測システム ===")
    
    # モデルの互換性を修正
    fix_success = fix_model_compatibility()
    
    # モデルを読み込む
    try:
        if fix_success and os.path.exists('fixed_rf_model.joblib'):
            # 修正したモデルを読み込む
            model = joblib.load('fixed_rf_model.joblib')
            print("修正したモデルを正常に読み込みました")
        else:
            # 元のモデルを読み込む
            model = joblib.load('fixed_rf_model.joblib')
            print("元のモデルを正常に読み込みました")
    except Exception as e:
        # 読み込みに失敗した場合はダミーモデルを作成
        print(f"モデルの読み込みに失敗しました: {e}")
        print("ダミーモデルを使用します")
        
        class DummyModel:
            def predict(self, X):
                return np.zeros(len(X))
        
        model = DummyModel()
    
    # 単一のテストデータで予測
    test_data = create_realistic_test_data()
    
    try:
        # 予測を実行
        prediction = model.predict(test_data)
        print("予測が成功しました")
        print(f"単一シナリオの予測結果: {prediction[0]:.1f}人の入院患者")
    except Exception as e:
        print(f"予測に失敗しました: {e}")
        print("ダミーの予測結果を使用します")
        prediction = np.zeros(len(test_data))
        print(f"単一シナリオの予測結果: {prediction[0]:.1f}人の入院患者")
    
    # 複数シナリオの予測
    print("\n複数シナリオの予測結果:")
    scenario_results = predict_multiple_scenarios(model)
    
    if len(scenario_results) > 0 and len(scenario_results[0]) == 3:
        # CSVデータからのシナリオの場合
        print("シナリオ, 予測値, 実測値, 誤差")
        for description, pred, actual in scenario_results:
            if isinstance(pred, str):
                print(f"- {description}: {pred} (実測値: {actual}人)")
            else:
                error = abs(pred - actual) if actual is not None else "N/A"
                print(f"- {description}: {pred:.1f}人の入院患者 (実測値: {actual}人, 誤差: {error})")
    else:
        # 手動作成シナリオの場合
        for description, pred in scenario_results:
            if isinstance(pred, str):
                print(f"- {description}: {pred}")
            else:
                print(f"- {description}: {pred:.1f}人の入院患者")
    

if __name__ == "__main__":
    main()