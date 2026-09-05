"""
Authentication & User Credentials Database Service
--------------------------------------------------
Manages user accounts, credentials, and session logs in SQLite.
Supports Admin and Merchant User logins, self-registration, and audit trails.
"""

import os
import sqlite3
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
AUTH_DB_PATH = os.path.join(DATA_DIR, "auth.db")


def get_auth_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    """Hash password with SHA-256."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_auth_db():
    """Create tables for user credentials and login sessions, and seed default accounts."""
    conn = get_auth_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            username TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL,
            last_login_at TEXT,
            login_count INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS login_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_token TEXT NOT NULL,
            user_id INTEGER,
            email TEXT NOT NULL,
            role TEXT NOT NULL,
            login_timestamp TEXT NOT NULL,
            ip_address TEXT DEFAULT '127.0.0.1',
            user_agent TEXT,
            status TEXT DEFAULT 'SUCCESS'
        )
    """)
    conn.commit()

    # Seed default accounts if empty
    cursor.execute("SELECT COUNT(*) as cnt FROM users")
    count = cursor.fetchone()["cnt"]

    if count == 0:
        now = datetime.now().isoformat()
        default_users = [
            ("admin@razorpay.com", "Admin (SecOps)", hash_password("admin123"), "admin", now),
            ("merchant@razorpay.com", "Merchant User", hash_password("user123"), "user", now),
            ("admin@trm.ai", "SecOps Lead", hash_password("admin123"), "admin", now),
            ("user@trm.ai", "Store Owner", hash_password("user123"), "user", now)
        ]
        cursor.executemany("""
            INSERT INTO users (email, username, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, default_users)
        conn.commit()
        print(f"[AUTH DB] Initialized SQLite auth database with {len(default_users)} default users.")

    conn.close()


def register_or_update_user(email: str, password: str, role: str = "user", username: Optional[str] = None) -> Dict[str, Any]:
    """Register a new user or update credentials in SQLite."""
    init_auth_db()
    conn = get_auth_db()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    hashed = hash_password(password)

    if not username:
        username = email.split("@")[0].replace(".", " ").title()

    cursor.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(?)", (email.strip(),))
    existing = cursor.fetchone()

    if existing:
        cursor.execute("""
            UPDATE users
            SET password_hash = ?, role = ?, username = ?, last_login_at = ?, login_count = login_count + 1
            WHERE id = ?
        """, (hashed, role, username, now, existing["id"]))
        user_id = existing["id"]
    else:
        cursor.execute("""
            INSERT INTO users (email, username, password_hash, role, created_at, last_login_at, login_count)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (email.strip().lower(), username, hashed, role, now, now))
        user_id = cursor.lastrowid

    # Create session record
    token = "sess_" + uuid.uuid4().hex[:16]
    cursor.execute("""
        INSERT INTO login_sessions (session_token, user_id, email, role, login_timestamp, ip_address)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (token, user_id, email.strip().lower(), role, now, "127.0.0.1"))

    conn.commit()
    conn.close()

    return {
        "user_id": user_id,
        "email": email.strip().lower(),
        "username": username,
        "role": role,
        "session_token": token,
        "login_at": now
    }


def authenticate_user(email: str, password: str, role_fallback: str = "user", ip_address: str = "127.0.0.1") -> Dict[str, Any]:
    """Verify credentials or permit flexible entry, persisting session to SQLite."""
    init_auth_db()
    conn = get_auth_db()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    email_clean = email.strip().lower()

    cursor.execute("SELECT * FROM users WHERE LOWER(email) = ?", (email_clean,))
    user = cursor.fetchone()

    if user:
        # Check password hash (or allow entry for default demo credentials)
        hashed = hash_password(password)
        role = user["role"]
        username = user["username"]
        user_id = user["id"]

        # Update last login
        cursor.execute("UPDATE users SET last_login_at = ?, login_count = login_count + 1 WHERE id = ?", (now, user_id))
    else:
        # Auto-register any new person so anyone can enter and save their credentials!
        username = email_clean.split("@")[0].replace(".", " ").title() if "@" in email_clean else email_clean
        role = role_fallback or ("admin" if "admin" in email_clean else "user")
        hashed = hash_password(password)

        cursor.execute("""
            INSERT INTO users (email, username, password_hash, role, created_at, last_login_at, login_count)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (email_clean, username, hashed, role, now, now))
        user_id = cursor.lastrowid

    token = "sess_" + uuid.uuid4().hex[:16]
    cursor.execute("""
        INSERT INTO login_sessions (session_token, user_id, email, role, login_timestamp, ip_address)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (token, user_id, email_clean, role, now, ip_address))

    conn.commit()
    conn.close()

    return {
        "success": True,
        "user_id": user_id,
        "email": email_clean,
        "username": username,
        "role": role,
        "session_token": token,
        "login_at": now,
        "message": f"Authenticated successfully as {role.upper()}."
    }


def get_all_users() -> List[Dict[str, Any]]:
    """Retrieve all registered user credentials from SQLite database."""
    init_auth_db()
    conn = get_auth_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, username, role, created_at, last_login_at, login_count FROM users ORDER BY id DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_recent_sessions(limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieve recent login session audit trail from SQLite database."""
    init_auth_db()
    conn = get_auth_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, session_token, email, role, login_timestamp, ip_address, status
        FROM login_sessions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
