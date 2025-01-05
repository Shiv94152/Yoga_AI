from flask import Blueprint, render_template
from flask_login import login_required, current_user
from datetime import datetime
from pymysql import func
from . import db

main = Blueprint('main', __name__)

@main.route('/')
@login_required
def dashboard():

    return render_template('index.html')