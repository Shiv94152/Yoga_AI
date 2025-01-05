# app/__init__.py
from flask import Flask
import pymysql
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# Configure PyMySQL to work with SQLAlchemy
pymysql.install_as_MySQLdb()

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'Admin123'  # Use a strong, unique secret key
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:shivsql123@localhost/yoga_ai'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    with app.app_context():
        from .models import User
        @login_manager.user_loader
        def load_user(user_id):
            return User.query.get(int(user_id))

        from .routes import main
        app.register_blueprint(main)

    return app

# app/models.py


# app/routes.py

# app/templates/dashboard.html

# requirements.txt


# .env
