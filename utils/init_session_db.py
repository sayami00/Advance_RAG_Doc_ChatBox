import sqlite3
from datetime import datetime
import os
from contextlib import contextmanager

DATABASE_PATH = "database/otp_sessions.db"
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    try:
        yield conn
    finally:
        conn.close()



def init_database():
    """Initialize SQLite database with required tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create OTP storage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS otp_storage (
                email TEXT PRIMARY KEY,
                otp TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                attempts INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_token TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                jwt_token TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # Create index for faster email lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_email ON user_sessions(email)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_active ON user_sessions(is_active, expires_at)
        """)
        
        conn.commit()

if __name__ == "__main__":
    print("🚀 Initializing session User Database...")
    init_database()
    print("\n📊 Database Status:")
