import random
import time
import smtplib
from flask import session

# Email Credentials and SMTP Configurations
SENDER_EMAIL = "omegaspecial3@gmail.com"
APP_PASSWORD = "tqqj kyww kvzt nrka"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def send_email(recipient_email, subject, body):
    try:
        # Sends an email using the pre-configured SMTP credentials.
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure the connection with TLS
            server.login(SENDER_EMAIL, APP_PASSWORD)
            message = f"Subject: {subject}\n\n{body}"
            server.sendmail(SENDER_EMAIL, recipient_email, message)
    except Exception as e:
        print("Error sendeing email: ",e)

def generate_otp():
    # Generates a random 6-digit OTP and calculates its expiration time (5 minutes ahead).
    otp = random.randint(100000, 999999)
    expiry = time.time() + 300  # OTP valid for 5 minutes
    return otp, expiry

def send_otp_via_email(email):
    # Generates an OTP, sends it to the provided email address, and stores the OTP and its expiry in the session.
    otp, expiry = generate_otp()
    subject = "Your OTP Code"
    body = f"Your OTP is {otp}. It is valid for 5 minutes."
    send_email(email, subject, body)
    
    # Store OTP details in the Flask session (temporary storage)
    session['otp'] = otp
    session['otp_expiry'] = expiry
    return otp

def verify_otp(user_input):
    # Verifies whether the user-provided OTP matches the one stored in the session and has not expired.    
    stored_otp = session.get('otp')
    otp_expiry = session.get('otp_expiry')
    
    # No OTP in session
    if stored_otp is None or otp_expiry is None:
        return False
    
    # Check if the OTP has expired
    if time.time() > otp_expiry:
        session.pop('otp', None)
        session.pop('otp_expiry', None)
        return False
    
    # Validate the OTP
    if str(stored_otp) == str(user_input):
        # Remove the OTP from session after successful verification to ensure one-time use
        session.pop('otp', None)
        session.pop('otp_expiry', None)
        return True
    
    return False