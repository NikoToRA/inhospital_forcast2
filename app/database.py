import sqlite3
import os
from datetime import datetime

def get_db_path():
    """データベースファイルのパスを取得"""
    return os.path.join(os.path.dirname(__file__), 'data', 'hospital_data.db')

def init_db():
    """データベースの初期化とテーブルの作成"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # training_dataテーブルの作成
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS training_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date DATE NOT NULL,
        total_outpatient INTEGER NOT NULL,
        intro_outpatient INTEGER NOT NULL,
        er_patients INTEGER NOT NULL,
        bed_count INTEGER NOT NULL,
        actual_inpatients INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # model_metricsテーブルの作成
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        mse REAL NOT NULL,
        mae REAL NOT NULL,
        r2_score REAL NOT NULL,
        evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 新規入院患者数の履歴データテーブルの作成
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS inpatient_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date DATE NOT NULL,
        new_inpatients INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()
    conn.close()

def add_training_data(date, total_outpatient, intro_outpatient, er_patients, bed_count, actual_inpatients):
    """トレーニングデータの追加"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO training_data (date, total_outpatient, intro_outpatient, er_patients, bed_count, actual_inpatients)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (date, total_outpatient, intro_outpatient, er_patients, bed_count, actual_inpatients))

    conn.commit()
    conn.close()

def get_training_data():
    """トレーニングデータの取得"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM training_data ORDER BY date')
    data = cursor.fetchall()

    conn.close()
    return data

def add_model_metrics(model_name, mse, mae, r2_score):
    """モデルの評価指標の保存"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO model_metrics (model_name, mse, mae, r2_score)
    VALUES (?, ?, ?, ?)
    ''', (model_name, mse, mae, r2_score))

    conn.commit()
    conn.close()

def get_model_metrics(model_name):
    """モデルの評価指標の取得"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM model_metrics WHERE model_name = ? ORDER BY evaluation_date DESC', (model_name,))
    metrics = cursor.fetchall()

    conn.close()
    return metrics

def add_inpatient_history(date, new_inpatients):
    """新規入院患者数の履歴データを追加"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO inpatient_history (date, new_inpatients)
    VALUES (?, ?)
    ''', (date, new_inpatients))

    conn.commit()
    conn.close()

def get_inpatient_history(start_date=None, end_date=None):
    """新規入院患者数の履歴データを取得"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = 'SELECT * FROM inpatient_history'
    params = []
    
    if start_date and end_date:
        query += ' WHERE date BETWEEN ? AND ?'
        params.extend([start_date, end_date])
    
    query += ' ORDER BY date'
    
    cursor.execute(query, params)
    data = cursor.fetchall()

    conn.close()
    return data 