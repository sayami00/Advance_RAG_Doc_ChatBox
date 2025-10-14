# otpoperation.py

import os
import jwt
import smtplib
import random
import string
import hashlib
import sqlite3
from typing import Dict, Optional
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import contextmanager


class OtpOperation:
    def __init__(self):
        # Configuration
        self.DATABASE_PATH = "database/otp_sessions.db"
        self.SECRET_KEY = "MyAIAPIkey"
        self.SESSION_EXPIRE_HOURS = 24
        self.ALGORITHM = "HS256"
        self.security = HTTPBearer()

        # Email config (optional if you implement `send_email_otp`)
        self.SMTP_SERVER = "smtp.gmail.com"
        self.SMTP_PORT = 587
        self.SENDER_EMAIL = "you@example.com"
        self.SENDER_PASSWORD = "yourpassword"  # use environment variables for security!

    @contextmanager
    def get_db_connection(self):
        conn = sqlite3.connect(self.DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def generate_otp(self, length: int = 6) -> str:
        return ''.join(random.choices(string.digits, k=length))

    def store_otp(self, email: str, otp: str, expires_at: datetime):
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO otp_storage (email, otp, expires_at, attempts)
                VALUES (?, ?, ?, 0)
            """, (email, otp, expires_at))
            conn.commit()

    def cleanup_expired_data(self):
        print("here i am ")
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            current_time = datetime.now()
            cursor.execute("DELETE FROM otp_storage WHERE expires_at < ?", (current_time,))
            cursor.execute("UPDATE user_sessions SET is_active = FALSE WHERE expires_at < ?", (current_time,))
            old_date = current_time - timedelta(days=7)
            cursor.execute("DELETE FROM user_sessions WHERE is_active = FALSE AND created_at < ?", (old_date,))
            conn.commit()

    def send_email_otp(self, email: str, otp: str) -> bool:
        try:
            msg = MIMEMultipart()
            msg['From'] = self.SENDER_EMAIL
            msg['To'] = email
            msg['Subject'] = "Your OTP Code"

            body = f"""
            Hello,

            Your One-Time Password (OTP) is: {otp}

            This OTP is valid for 5 minutes only.

            If you didn't request this OTP, please ignore this email.

            Best regards,
            Your App Team
            """
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.SMTP_SERVER, self.SMTP_PORT) as server:
                server.starttls()
                server.login(self.SENDER_EMAIL, self.SENDER_PASSWORD)
                server.send_message(msg)

            return True
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return False

    def get_otp_data(self, email: str) -> Optional[Dict]:
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT otp, expires_at, attempts FROM otp_storage WHERE email = ?", (email,))
            row = cursor.fetchone()
            if row:
                return {
                    "otp": row["otp"],
                    "expires_at": datetime.fromisoformat(row["expires_at"]),
                    "attempts": row["attempts"]
                }
            return None

    def delete_otp(self, email: str):
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM otp_storage WHERE email = ?", (email,))
            conn.commit()

    def create_jwt_token(self, email: str, session_token: str) -> str:
        payload = {
            "email": email,
            "session_token": session_token,
            "exp": datetime.utcnow() + timedelta(hours=self.SESSION_EXPIRE_HOURS),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.SECRET_KEY, algorithm=self.ALGORITHM)

    def generate_session_token(self) -> str:
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
        timestamp = str(datetime.now().timestamp())
        return hashlib.sha256((random_string + timestamp).encode()).hexdigest()

    def create_user_session_db(self, email: str) -> Dict:
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE user_sessions SET is_active = FALSE WHERE email = ?", (email,))
            session_token = self.generate_session_token()
            expires_at = datetime.now() + timedelta(hours=self.SESSION_EXPIRE_HOURS)
            jwt_token = self.create_jwt_token(email, session_token)
            cursor.execute("""
                INSERT INTO user_sessions (session_token, email, jwt_token, expires_at, is_active)
                VALUES (?, ?, ?, ?, TRUE)
            """, (session_token, email, jwt_token, expires_at))
            conn.commit()

            return {
                "email": email,
                "session_token": session_token,
                "jwt_token": jwt_token,
                "created_at": datetime.now(),
                "expires_at": expires_at,
                "is_active": True
            }

    def increment_otp_attempts(self, email: str) -> int:
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE otp_storage SET attempts = attempts + 1 WHERE email = ?", (email,))
            cursor.execute("SELECT attempts FROM otp_storage WHERE email = ?", (email,))
            row = cursor.fetchone()
            conn.commit()
            return row["attempts"] if row else 0

    def verify_jwt_token(self, token: str) -> Dict:
        try:
            payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    def get_session_by_token(self, session_token: str) -> Optional[Dict]:
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_token, email, jwt_token, created_at, expires_at, is_active
                FROM user_sessions 
                WHERE session_token = ?
            """, (session_token,))
            row = cursor.fetchone()
            if row:
                return {
                    "session_token": row["session_token"],
                    "email": row["email"],
                    "jwt_token": row["jwt_token"],
                    "created_at": datetime.fromisoformat(row["created_at"]),
                    "expires_at": datetime.fromisoformat(row["expires_at"]),
                    "is_active": bool(row["is_active"])
                }
            return None

    def invalidate_session(self, session_token: str):
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE user_sessions SET is_active = FALSE WHERE session_token = ?", (session_token,))
            conn.commit()

    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> Dict:
        token = credentials.credentials
        payload = self.verify_jwt_token(token)
        session_token = payload.get("session_token")
        email = payload.get("email")

        session_data = self.get_session_by_token(session_token)
        if not session_data:
            raise HTTPException(status_code=401, detail="Session not found")
        if not session_data["is_active"]:
            raise HTTPException(status_code=401, detail="Session is inactive")
        if datetime.now() > session_data["expires_at"]:
            self.invalidate_session(session_token)
            raise HTTPException(status_code=401, detail="Session has expired")
        if session_data["email"] != email:
            raise HTTPException(status_code=401, detail="Invalid session")
        return session_data

    def get_all_active_sessions(self) -> list:
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT email, created_at, expires_at
                FROM user_sessions 
                WHERE is_active = TRUE AND expires_at > ?
                ORDER BY created_at DESC
            """, (datetime.now(),))
            return [
                {
                    "email": row["email"],
                    "created_at": row["created_at"],
                    "expires_at": row["expires_at"]
                }
                for row in cursor.fetchall()
            ]
