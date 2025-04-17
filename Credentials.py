import sqlite3
import bcrypt
import datetime
from flask import g

DB_NAME = "users.db"

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_NAME)
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_auth_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS Users(
                       id TEXT PRIMARY KEY,
                       username TEXT UNIQUE,
                       email TEXT UNIQUE,
                       password TEXT,
                       verified INTEGER DEFAULT 0)''')
        conn.commit()

def generateUserId():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        year = datetime.datetime.now().year
        month = datetime.datetime.now().month

        cursor.execute("SELECT id FROM Users ORDER BY id DESC LIMIT 1")
        last_id = cursor.fetchone()

    if last_id:
        try:
            last_number = int(last_id[0].split('-')[-1]) + 1
        except (ValueError, IndexError):
            last_number = 0
    else:
        last_number = 0
    
    return f"GAAN-{year}-{month:02d}-{last_number:03d}"

def register_user(username, email, password):
    try:
        db = get_db()  # Use Flask's context to get db
        cursor = db.cursor()
        new_id = generateUserId()
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        cursor.execute(
            "INSERT INTO Users (id, username, email, password) VALUES (?, ?, ?, ?)",
            (new_id, username, email, hashed_password)
        )
        db.commit()
        return "User registered successfully"
    except sqlite3.IntegrityError:
        return "Username or email already registered"
    finally:
        close_db()  # Ensure db is closed after use

def login_user(username, password):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT password FROM Users WHERE username = ?", (username,))
    result = cursor.fetchone()
    close_db()  # Close db after query
    
    if result:
        stored_password = result[0]
        if bcrypt.checkpw(password.encode(), stored_password.encode()):
            return "Login successful"
        return "Incorrect password"
    return "User not found"

# Initialize database on first import
init_auth_db()