from flask import Flask
import os

def create_app():
    # テンプレートとスタティックディレクトリのパスを明示的に指定
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
    
    app = Flask(__name__, 
                template_folder=template_dir,
                static_folder=static_dir)
    
    # 基本設定
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev')
    app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # ブループリントの登録
    from app.routes import main
    app.register_blueprint(main)
    
    return app 