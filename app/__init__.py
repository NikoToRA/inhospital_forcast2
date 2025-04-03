from flask import Flask
from config import Config
import os

def check_environment():
    """環境変数の存在を確認"""
    missing_vars = []
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if not supabase_url:
        missing_vars.append('SUPABASE_URL')
    if not supabase_key:
        missing_vars.append('SUPABASE_KEY')
        
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    print("All required environment variables are set")
    return supabase_url, supabase_key

def create_app():
    """アプリケーションファクトリ"""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # 環境変数の確認
    supabase_url, supabase_key = check_environment()
    
    # テンプレートとスタティックディレクトリのパスを明示的に指定
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
    
    app.config['SUPABASE_URL'] = supabase_url
    app.config['SUPABASE_KEY'] = supabase_key
    
    # ブループリントの登録
    from app.routes import main
    app.register_blueprint(main)
    
    return app

# アプリケーションインスタンスを作成
app = create_app() 