-- hospital_dataテーブルの作成
CREATE TABLE IF NOT EXISTS hospital_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    total_outpatient INTEGER NOT NULL,
    intro_outpatient INTEGER NOT NULL,
    er_patients INTEGER NOT NULL,
    bed_count INTEGER NOT NULL,
    y INTEGER NOT NULL,  -- 新入院患者数
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

-- インデックスの作成
CREATE INDEX IF NOT EXISTS idx_hospital_data_date ON hospital_data(date);

-- training_dataテーブルの作成（既存のテーブル）
CREATE TABLE IF NOT EXISTS training_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    mon BOOLEAN NOT NULL,
    tue BOOLEAN NOT NULL,
    wed BOOLEAN NOT NULL,
    thu BOOLEAN NOT NULL,
    fri BOOLEAN NOT NULL,
    sat BOOLEAN NOT NULL,
    sun BOOLEAN NOT NULL,
    public_holiday BOOLEAN NOT NULL,
    public_holiday_previous_day BOOLEAN NOT NULL,
    total_outpatient INTEGER NOT NULL,
    intro_outpatient INTEGER NOT NULL,
    er INTEGER NOT NULL,
    bed_count INTEGER NOT NULL,
    y FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

-- インデックスの作成
CREATE INDEX IF NOT EXISTS idx_training_data_date ON training_data(date); 