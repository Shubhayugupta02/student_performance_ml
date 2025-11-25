import os
import sqlite3


DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")


def create_database():
    """Create users.db and the users table (if not already there)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT CHECK(role IN ('teacher','student')) NOT NULL
        );
        """
    )

    conn.commit()
    conn.close()
    print("[OK] Database and users table ready.")


def insert_bulk_users(num_teachers: int = 200, num_students: int = 2000):
    """
    Insert many teacher_i / student_j users.
    Teachers:  teacher1 ... teacher<num_teachers>
    Students:  student1 ... student<num_students>
    Passwords: teacher<i>@123456789 and student<j>@123456789
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    inserted = 0

    # teachers
    for i in range(1, num_teachers + 1):
        username = f"teacher{i}"
        password = f"teacher{i}@123456789"
        try:
            cur.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, password, "teacher"),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            # already exists, skip
            continue

    # students
    for j in range(1, num_students + 1):
        username = f"student{j}"
        password = f"student{j}@123456789"
        try:
            cur.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, password, "student"),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            continue

    conn.commit()
    conn.close()
    print(f"[OK] Inserted/kept {inserted} users (teachers + students).")


if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("[INFO] Old users.db removed.")

    create_database()
    insert_bulk_users()
    print("[DONE] users.db created with many demo users.")







