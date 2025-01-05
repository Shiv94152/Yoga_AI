
from Flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
from werkzeug.security import generate_password_hash
import re
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Database path
DB_PATH = 'users.db'

class DatabaseError(Exception):
    pass

def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {str(e)}")
        raise DatabaseError("Could not connect to database")

def init_db():
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS users
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         fullname TEXT NOT NULL,
                         email TEXT UNIQUE NOT NULL,
                         password TEXT NOT NULL,
                         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            conn.commit()
            logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Database initialization error: {str(e)}")
        raise

@app.before_first_request
def setup():
    try:
        init_db()
    except Exception as e:
        logging.error(f"Setup error: {str(e)}")

def validate_input(fullname, email, password, confirm_password):
    errors = []
    
    if not fullname or len(fullname.strip()) < 2:
        errors.append("Full name must be at least 2 characters long")
    
    email_pattern = r"[^@]+@[^@]+\.[^@]+"
    if not re.match(email_pattern, email):
        errors.append("Invalid email format")
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    if password != confirm_password:
        errors.append("Passwords do not match")
    
    return errors

@app.route('/')
def index():
    return render_template('index.html', debug=app.debug)

@app.route('/signup', methods=['POST'])
def signup():
    try:
        # Log form submission
        logging.info("Received signup form submission")
        
        # Get form data
        fullname = request.form.get('fullname', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Log received data (exclude passwords)
        logging.debug(f"Received signup data - Name: {fullname}, Email: {email}")
        
        # Validate input
        validation_errors = validate_input(fullname, email, password, confirm_password)
        
        if validation_errors:
            for error in validation_errors:
                flash(error, 'error')
                logging.warning(f"Validation error: {error}")
            return redirect(url_for('index'))
        
        # Try to create user
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                
                # Check for existing email
                c.execute('SELECT email FROM users WHERE email = ?', (email,))
                if c.fetchone():
                    flash('Email already registered', 'error')
                    logging.warning(f"Duplicate email attempt: {email}")
                    return redirect(url_for('index'))
                
                # Hash password and save user
                hashed_password = generate_password_hash(password)
                c.execute('''
                    INSERT INTO users (fullname, email, password, created_at) 
                    VALUES (?, ?, ?, ?)
                ''', (fullname, email, hashed_password, datetime.now()))
                conn.commit()
                
                logging.info(f"New user created successfully: {email}")
                flash('Account created successfully! Please check your email for verification.', 'success')
                
        except sqlite3.Error as e:
            logging.error(f"Database error during user creation: {str(e)}")
            flash('An error occurred while creating your account. Please try again.', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        logging.error(f"Unexpected error in signup: {str(e)}")
        flash('An unexpected error occurred. Please try again.', 'error')
        
    return redirect(url_for('index'))

@app.route('/debug/users')
def debug_users():
    if not app.debug:
        return "Debug mode is not enabled", 403
        
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT id, fullname, email, created_at FROM users')
            users = c.fetchall()
            return render_template('debug.html', users=users)
    except Exception as e:
        logging.error(f"Error in debug view: {str(e)}")
        return str(e), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    logging.error(f"404 error: {request.url}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"500 error: {str(error)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)