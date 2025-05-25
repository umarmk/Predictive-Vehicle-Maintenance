"""
Authentication module for the Predictive Vehicle Maintenance System.
Provides user registration, login, and session management functionality.
"""

import hashlib
import secrets
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json

class AuthManager:
    def __init__(self, db_path: str = "users.db"):
        """Initialize the authentication manager with SQLite database."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the user database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash a password with salt."""
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 for secure password hashing
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 100,000 iterations
        ).hex()
        
        return password_hash, salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify a password against its hash."""
        computed_hash, _ = self.hash_password(password, salt)
        return computed_hash == password_hash
    
    def register_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Register a new user."""
        try:
            # Validate input
            if not username or not email or not password:
                return {"success": False, "error": "All fields are required"}
            
            if len(password) < 6:
                return {"success": False, "error": "Password must be at least 6 characters long"}
            
            # Hash password
            password_hash, salt = self.hash_password(password)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash, salt)
                    VALUES (?, ?, ?, ?)
                ''', (username, email, password_hash, salt))
                
                user_id = cursor.lastrowid
                conn.commit()
                
                return {
                    "success": True,
                    "message": "User registered successfully",
                    "user_id": user_id
                }
                
            except sqlite3.IntegrityError as e:
                if "username" in str(e):
                    return {"success": False, "error": "Username already exists"}
                elif "email" in str(e):
                    return {"success": False, "error": "Email already exists"}
                else:
                    return {"success": False, "error": "Registration failed"}
            
            finally:
                conn.close()
                
        except Exception as e:
            return {"success": False, "error": f"Registration failed: {str(e)}"}
    
    def login_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate a user and create a session."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user by username or email
            cursor.execute('''
                SELECT id, username, email, password_hash, salt, is_active
                FROM users
                WHERE (username = ? OR email = ?) AND is_active = 1
            ''', (username, username))
            
            user = cursor.fetchone()
            
            if not user:
                return {"success": False, "error": "Invalid username or password"}
            
            user_id, db_username, email, password_hash, salt, is_active = user
            
            # Verify password
            if not self.verify_password(password, password_hash, salt):
                return {"success": False, "error": "Invalid username or password"}
            
            # Create session token
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(days=7)  # 7 days expiry
            
            # Store session
            cursor.execute('''
                INSERT INTO sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_id, session_token, expires_at))
            
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "message": "Login successful",
                "session_token": session_token,
                "user": {
                    "id": user_id,
                    "username": db_username,
                    "email": email
                }
            }
            
        except Exception as e:
            return {"success": False, "error": f"Login failed: {str(e)}"}
    
    def verify_session(self, session_token: str) -> Dict[str, Any]:
        """Verify a session token and return user information."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.user_id, s.expires_at, u.username, u.email
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = ? AND s.is_active = 1 AND u.is_active = 1
            ''', (session_token,))
            
            session = cursor.fetchone()
            conn.close()
            
            if not session:
                return {"success": False, "error": "Invalid session"}
            
            user_id, expires_at, username, email = session
            
            # Check if session is expired
            if datetime.fromisoformat(expires_at) < datetime.now():
                self.logout_user(session_token)
                return {"success": False, "error": "Session expired"}
            
            return {
                "success": True,
                "user": {
                    "id": user_id,
                    "username": username,
                    "email": email
                }
            }
            
        except Exception as e:
            return {"success": False, "error": f"Session verification failed: {str(e)}"}
    
    def logout_user(self, session_token: str) -> Dict[str, Any]:
        """Logout a user by invalidating their session."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE sessions SET is_active = 0 WHERE session_token = ?
            ''', (session_token,))
            
            conn.commit()
            conn.close()
            
            return {"success": True, "message": "Logout successful"}
            
        except Exception as e:
            return {"success": False, "error": f"Logout failed: {str(e)}"}
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Session cleanup failed: {str(e)}")

# Global auth manager instance
auth_manager = AuthManager()
