# init_db.py
"""
Database initialization script for Chatbox User Registration
Run this script to create the database and tables
"""

import sqlite3
from datetime import datetime
import os

def init_database():
    """Initialize the SQLite database with required tables"""
    
    # Database file path
    db_path = "database/chatbox_users.db"
    
    # Remove existing database if it exists (for fresh start)
    if os.path.exists(db_path):
        print(f"Removing existing database: {db_path}")
        os.remove(db_path)
    
    # Create connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"Creating database: {db_path}")
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email VARCHAR(255) UNIQUE NOT NULL,
            department VARCHAR(255) NOT NULL,
            username VARCHAR(255),
            is_active BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index for faster email lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_department ON users(department)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active)")
    
    # Insert some sample data for testing
    sample_users = [
        ("john.doe@company.com", "Engineering", "johndoe"),
        ("jane.smith@company.com", "Marketing", "janesmith"),
        ("admin@company.com", "IT", "admin"),
        ("demo@example.com", "Sales", "demo"),
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO users (email, department, username) VALUES (?, ?, ?)",
        sample_users
    )
    
    # Commit changes and close
    conn.commit()
    
    # Sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            email TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            is_active BOOLEAN DEFAULT TRUE
        )
    ''')
    
    # Conversations table (for mapping chat_id to conversation_id)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT UNIQUE NOT NULL,
            user_name TEXT NOT NULL,
            session_id TEXT NOT NULL,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Conversations_old table (for mapping chat_id to conversation_id)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations_old (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT UNIQUE NOT NULL,
            conversation_id TEXT UNIQUE NOT NULL,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    ''')


    # Documents table for RAG
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            uploaded_by TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed BOOLEAN DEFAULT FALSE
        )
    ''')
    
    conn.commit()

    # Display created tables and data
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Created tables: {[table[0] for table in tables]}")
    
    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    print(f"Sample users created: {user_count}")
    
    cursor.execute("SELECT email, department, username FROM users")
    users = cursor.fetchall()
    print("\nSample users:")
    for user in users:
        print(f"  - {user[0]} ({user[1]}) - {user[2]}")
    
    conn.close()
    print(f"\nDatabase initialization complete! ✅")
    print(f"Database file: {os.path.abspath(db_path)}")

def check_database():
    """Check database status and contents"""
    db_path = "database/chatbox_users.db"
    
    if not os.path.exists(db_path):
        print("❌ Database file not found!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check table structure
    cursor.execute("PRAGMA table_info(users)")
    columns = cursor.fetchall()
    print("Users table structure:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    # Check data
    cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
    active_users = cursor.fetchone()[0]
    
    cursor.execute("SELECT department, COUNT(*) FROM users WHERE is_active = 1 GROUP BY department")
    dept_counts = cursor.fetchall()
    
    print(f"\nActive users: {active_users}")
    print("Users by department:")
    for dept, count in dept_counts:
        print(f"  - {dept}: {count}")
    
    conn.close()

if __name__ == "__main__":
    print("🚀 Initializing Chatbox User Database...")
    init_database()
    print("\n📊 Database Status:")
    check_database()
