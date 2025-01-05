# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pymysql
from werkzeug.security import generate_password_hash, check_password_hash
import re
from datetime import datetime
import subprocess
from flask import jsonify
import sys
import os
#import base64
#from flask_cors import CORS
#import mediapipe as mp

app = Flask(__name__)
app.secret_key = 'Admin123'

# Database connection function
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='shivsql123',
        database='yoga_ai',
        cursorclass=pymysql.cursors.DictCursor
    )

@app.route('/')
def home():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    with conn.cursor() as cursor:
        # Get user info
        cursor.execute('SELECT * FROM users WHERE id = %s', (session['id'],))
        user = cursor.fetchone()
        
        # Get practice statistics
        cursor.execute('''
            SELECT COUNT(*) as total_sessions, 
            SUM(duration_minutes) as total_minutes 
            FROM practice_sessions 
            WHERE user_id = %s AND 
            MONTH(session_date) = MONTH(CURRENT_DATE())
        ''', (session['id'],))
        stats = cursor.fetchone()
        
        # Get upcoming classes
        cursor.execute('''
            SELECT * FROM yoga_classes 
            WHERE start_time > NOW() 
            ORDER BY start_time 
            LIMIT 3
        ''')
        classes = cursor.fetchall()
        
    conn.close()
    
    return render_template('index.html', 
                         user=user, 
                         stats=stats, 
                         classes=classes)
    

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        passwd = request.form['password']
        
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute('SELECT * FROM users WHERE email = %s', (email))
            user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], passwd):
            session['loggedin'] = True
            session['id'] = user['id']
            session['email'] = user['email']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email/password!', 'error')
    
    return render_template('auth.html', mode='login')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        password = request.form['password']
        
        if not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash('Invalid email address!', 'error')
            return render_template('auth.html', mode='signup')
            
        if not full_name or not password:
            flash('Please fill out all fields!', 'error')
            return render_template('auth.html', mode='signup')
        
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Check if email exists
            cursor.execute('SELECT id FROM users WHERE email = %s', (email,))
            if cursor.fetchone():
                flash('Account already exists with this email!', 'error')
                return render_template('auth.html', mode='signup')
            
            # Insert new user
            hashed_password = generate_password_hash(password)
            cursor.execute('''
                INSERT INTO users (full_name, email, password, created_at) 
                VALUES (%s, %s, %s, %s)
            ''', (full_name, email, hashed_password, datetime.now()))
            conn.commit()
            
        conn.close()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('auth.html', mode='signup')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/track_practice', methods=['POST'])
def track_practice():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
        
    duration = request.form.get('duration', type=int)
    if not duration:
        flash('Please enter valid duration', 'error')
        return redirect(url_for('home'))
        
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute('''
            INSERT INTO practice_sessions (user_id, duration_minutes, session_date) 
            VALUES (%s, %s, NOW())
        ''', (session['id'], duration))
        conn.commit()
    conn.close()
    
    flash('Practice session recorded!', 'success')
    return redirect(url_for('home'))
@app.route('/try_now')
def try_now():
    return render_template('CameraButton.html')



@app.route('/About')
def About():
    return render_template('About.html')




@app.route('/watch')
def Watch():
    return render_template('Video.html')

@app.route('/Camera')
def StartCamera():
    return render_template('CameraButton.html')  # Ensure this matches your HTML filename

# Route to start the camera and execute the Python script
@app.route('/start_camera', methods=['POST'])
def start_camera():
    try:
        # Get the path to MediaPipeSetCounter.py
        script_path = os.path.join(os.path.dirname(__file__), 'MediaPipeSetCounter.py')
        
        # Start the MediaPipeSetCounter.py script
        process = subprocess.Popen([sys.executable, script_path])
        
        return jsonify({
            'status': 'success',
            'message': 'Camera started successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/execute/<program>', methods=['POST'])
def execute_program(program):
    try:
        # Define the script paths
        base_path = os.path.dirname(__file__)
        scripts = {
            'yoga': os.path.join(base_path, 'RealTimePoseDetection.py'),
            'counter': os.path.join(base_path, 'MediaPipeSetCounter.py')
        }
        
        if program not in scripts:
            return jsonify({
                'status': 'error',
                'message': 'Invalid program specified'
            }), 400
            
        # Execute the selected program
        script_path = scripts[program]
        process = subprocess.Popen([sys.executable, script_path])
        
        return jsonify({
            'status': 'success',
            'message': f'{program.title()} program started successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500








# MySQL setup queries
SETUP_QUERIES = [
  #  """
   # CREATE TABLE IF NOT EXISTS users (
    #    id INT AUTO_INCREMENT PRIMARY KEY,
     #   full_name VARCHAR(100) NOT NULL,
      #  email VARCHAR(100) UNIQUE NOT NULL,
       # password VARCHAR(255) NOT NULL,
        #created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    #)
    #""",
    """
    CREATE TABLE IF NOT EXISTS practice_sessions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        duration_minutes INT NOT NULL,
        session_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS yoga_classes (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        instructor VARCHAR(100) NOT NULL,
        start_time DATETIME NOT NULL,
        duration_minutes INT NOT NULL,
        class_type VARCHAR(50) NOT NULL
    )
    """
]

def init_db():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        for query in SETUP_QUERIES:
            cursor.execute(query)
    conn.commit()
    conn.close()
    app.run(debug=True)
    
#def setCamera():
if __name__ == '__main__':
    init_db()  # Initialize database tables
    app.run(debug=True)




