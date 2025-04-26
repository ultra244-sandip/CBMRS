import sqlite3
import bcrypt
import datetime

from flask import g
from auth import send_otp_via_email

DB_NAME = "users.db"

# Get DB connection using Flask's g context
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_NAME)
        g.db.row_factory = sqlite3.Row
    return g.db

# Close DB connection
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Initialize DB and create the updated Users table
def init_auth_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS Users(
                            id TEXT PRIMARY KEY,
                            username TEXT UNIQUE,
                            email TEXT UNIQUE,
                            password TEXT,
                            verified INTEGER DEFAULT 0,
                            subscription TEXT DEFAULT 'Regular')''')
        conn.commit()

# Generate a new user ID
def generateUserId():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        year = datetime.datetime.now().year
        month = datetime.datetime.now().month

        cursor.execute("SELECT id FROM Users ORDER BY id DESC LIMIT 1")
        last_id = cursor.fetchone()

    if last_id:
        try:
            parts = last_id[0].split('-')
            last_month = int(parts[2])
            last_number = int(parts[3])

            # Reset if new month or number > 999
            if month != last_month or last_number > 999:
                last_number = 0
            else:
                last_number += 1
        except (ValueError, IndexError):
            last_number = 0
    else:
        last_number = 0

    return f"GAAN-{year}-{month:02d}-{last_number:03d}"


# Register a new user, with optional subscription (default = 'Regular')
def register_user(username, email, password, final=False):
    try:
        db = get_db()
        cursor = db.cursor()
        new_id = generateUserId()
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        verified_final = 1 if final else 0

        cursor.execute(
            """INSERT INTO Users (id, username, email, password, verified)
               VALUES (?, ?, ?, ?, ?)""",
            (new_id, username, email, hashed_password, verified_final)
        )
        db.commit()
        
        if not final:
            send_otp_via_email(email)
            return "Registration initiated. Please check your email for OTP."
        else:
            return "Registration coplete. Email verified successfully."

    except sqlite3.IntegrityError:
        return "Username or email already registered"
    finally:
        close_db()


# Login user and check password
def login_user(username, password):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT password, verified, email FROM Users WHERE username = ?", (username,))
    result = cursor.fetchone()
    close_db()

    if result:
        stored_password, verified, email = result
        if bcrypt.checkpw(password.encode(), stored_password.encode()):
            if verified == 0:
                return {
                    "success" : False,
                    "message" : "Email not verified",
                    "email" : email
                }
            return "Login successful"
        return "Incorrect password"
    return "User not found"

# Initialize DB on first import
init_auth_db()
