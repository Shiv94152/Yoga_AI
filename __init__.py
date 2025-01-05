# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:shivsql123@localhost/yoga_ai'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    from .models import User
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    from .routes import main
    app.register_blueprint(main)

    return app

# app/models.py
from . import db
from flask_login import UserMixin
from datetime import datetime

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    practice_sessions = db.relationship('PracticeSession', backref='user', lazy=True)
    mastered_poses = db.relationship('MasteredPose', backref='user', lazy=True)

class PracticeSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    duration_minutes = db.Column(db.Integer, nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class MasteredPose(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    pose_name = db.Column(db.String(100), nullable=False)
    mastered_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class YogaClass(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    instructor = db.Column(db.String(100), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False)
    duration_minutes = db.Column(db.Integer, nullable=False)
    class_type = db.Column(db.String(50), nullable=False)

# app/routes.py
from flask import Blueprint, render_template
from flask_login import login_required, current_user
from .models import PracticeSession, MasteredPose, YogaClass
from datetime import datetime
from sqlalchemy import func
from . import db

main = Blueprint('main', __name__)

@main.route('/')
@login_required
def dashboard():
    # Get practice minutes for current month
    current_month = datetime.utcnow().month
    practice_minutes = db.session.query(func.sum(PracticeSession.duration_minutes))\
        .filter(PracticeSession.user_id == current_user.id)\
        .filter(func.extract('month', PracticeSession.date) == current_month)\
        .scalar() or 0

    # Get mastered poses count
    poses_mastered = MasteredPose.query.filter_by(user_id=current_user.id).count()

    # Get upcoming classes
    upcoming_classes = YogaClass.query\
        .filter(YogaClass.start_time > datetime.utcnow())\
        .order_by(YogaClass.start_time)\
        .limit(3)\
        .all()

    return render_template('dashboard.py',
                         practice_minutes=practice_minutes,
                         poses_mastered=poses_mastered,
                         upcoming_classes=upcoming_classes)

# app/templates/dashboard.html

# requirements.txt
flask==2.0.1
flask-sqlalchemy==2.5.1
flask-login==0.5.0
mysqlclient==2.0.3
python-dotenv==0.19.0

# .env
FLASK_APP=app
FLASK_ENV=development
DATABASE_URL=mysql://root:shivsql123@localhost/yoga_ai_db
SECRET_KEY=your-secret-key