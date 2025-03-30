from app import create_app
import os

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '127.0.0.1')  # localhost
    app.run(debug=app.config['DEBUG'], host=host, port=port) 