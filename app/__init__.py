from flask import Flask
from config import Config, SUPABASE_URL, SUPABASE_KEY
import os

def check_environment():
    """環境変数の存在を確認"""
    missing_vars = []
    
    if not SUPABASE_URL:
        missing_vars.append('SUPABASE_URL')
    else:
        print(f"SUPABASE_URL is set (length: {len(SUPABASE_URL)})")
        
    if not SUPABASE_KEY:
        missing_vars.append('SUPABASE_KEY')
    else:
        print(f"SUPABASE_KEY is set (length: {len(SUPABASE_KEY)})")
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    print("All required environment variables are set")

def create_app():
    """アプリケーションファクトリ"""
    # 環境変数の確認
    print("\nChecking environment variables...")
    check_environment()
    
    # テンプレートとスタティックディレクトリのパスを明示的に指定
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
    
    app = Flask(__name__, 
                template_folder=template_dir,
                static_folder=static_dir)
    
    # 基本設定
    app.config.from_object(Config)
    
    # ブループリントの登録
    from app.routes import main
    app.register_blueprint(main)
    
    return app 