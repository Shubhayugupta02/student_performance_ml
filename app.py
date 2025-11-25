# ================================================================
# CYBERML – STUDENT PERFORMANCE PORTAL (Render-ready app.py)
#   - Login (SQLite users.db)
#   - Dataset upload + ML training
#   - Prediction form (separate, no performance charts)
#   - Performance dashboard JSON APIs for Chart.js (bar/pie/regression/heatmap)
#   - Teacher / Student dashboards
#   - Teacher assignments + student submissions + grading + plagiarism
#   - Leaderboard
#   - Online IDE (Python only)
#
#   Compatible with:
#       - Local:  python app.py
#       - Render: gunicorn app:app
# ================================================================

import os
import json
import sqlite3
import shutil
import subprocess
from datetime import datetime, date

import numpy as np
import pandas as pd
import joblib

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    session,
    send_file,
    send_from_directory,
    abort,
)
from werkzeug.utils import secure_filename
from functools import wraps

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")  # important for Render / servers without display
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================
# BASIC FLASK / PATH SETUP
# ================================================================

app = Flask(__name__)
app.secret_key = "super_strong_key_change_me"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DB_PATH = os.path.join(BASE_DIR, "users.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
CHART_FOLDER = os.path.join(BASE_DIR, "static", "charts")
ASSIGNMENT_FOLDER = os.path.join(BASE_DIR, "assignments")
SUBMISSION_FOLDER = os.path.join(BASE_DIR, "submissions")
DATA_FOLDER = os.path.join(BASE_DIR, "data")

ASSIGNMENT_META = os.path.join(DATA_FOLDER, "assignments.json")
GRADES_META = os.path.join(DATA_FOLDER, "grades.json")

for folder in [
    UPLOAD_FOLDER,
    MODEL_FOLDER,
    CHART_FOLDER,
    ASSIGNMENT_FOLDER,
    SUBMISSION_FOLDER,
    DATA_FOLDER,
]:
    os.makedirs(folder, exist_ok=True)

# global ML state
current_df: pd.DataFrame | None = None
best_model_name: str | None = None
features_global: list[str] = []
scores_global: dict[str, float] = {}
feature_importances_global: dict[str, float] | None = None
target_col_global: str | None = None

# ================================================================
# DB HELPERS
# ================================================================

def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_user(username: str, password: str):
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT username, role FROM users WHERE username=? AND password=?",
            (username, password),
        )
        row = cur.fetchone()
        conn.close()
        return row
    except Exception as e:
        print("DB error:", e)
        return None


# ================================================================
# AUTH DECORATOR
# ================================================================

def login_required(role: str | None = None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if "username" not in session:
                return redirect(url_for("login"))
            if role and session.get("role") != role:
                # wrong role – send back home
                return redirect(url_for("home"))
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ================================================================
# JSON META HELPERS (ASSIGNMENTS + GRADES)
# ================================================================

def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_assignments():
    return _load_json(ASSIGNMENT_META, [])


def save_assignments(items):
    _save_json(ASSIGNMENT_META, items)


def load_grades():
    return _load_json(GRADES_META, [])


def save_grades(items):
    _save_json(GRADES_META, items)


def get_grade_for_file(filename: str):
    for g in load_grades():
        if g.get("filename") == filename:
            return g
    return None


# ================================================================
# ML HELPERS
# ================================================================

def detect_target(df: pd.DataFrame) -> str:
    """Guess the target column for marks."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric:
        raise ValueError("No numeric columns in dataset")

    # Try some common names
    for c in numeric:
        low = c.lower()
        if "final" in low or "total" in low or "marks" in low:
            return c
    for c in numeric:
        low = c.lower()
        if "percent" in low or "score" in low or "result" in low:
            return c

    # Fallback – choose last numeric column
    return numeric[-1]


def choose_features(df: pd.DataFrame, target: str):
    preferred = [
        "prev_percent",
        "internal_marks",
        "attendance",
        "study_hours",
        "assignments",
    ]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in preferred if c in numeric and c != target]
    if feats:
        return feats
    return [c for c in numeric if c != target]


def grade_of(v: float) -> str:
    if v >= 85:
        return "A"
    if v >= 70:
        return "B"
    if v >= 55:
        return "C"
    if v >= 40:
        return "D"
    return "F"


def safe_savefig(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
        plt.savefig(path, bbox_inches="tight", dpi=120)
    except Exception as e:
        print("savefig error:", e)
    finally:
        plt.close()


def generate_charts(df: pd.DataFrame, target: str, features: list[str]):
    """Generate static PNG charts for performance page."""
    sns.set_theme(style="darkgrid")

    # correlation heatmap
    try:
        corr = df[features + [target]].corr(numeric_only=True)
        plt.figure(figsize=(5, 4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="rocket_r")
        plt.title("Correlation heatmap")
        safe_savefig(os.path.join(CHART_FOLDER, "heatmap.png"))
    except Exception as e:
        print("heatmap error:", e)

    # regression scatter (first feature)
    try:
        main = features[0]
        plt.figure(figsize=(5, 4))
        sns.scatterplot(x=df[main], y=df[target])
        sns.regplot(x=df[main], y=df[target], scatter=False, color="cyan")
        plt.xlabel(main)
        plt.ylabel(target)
        plt.title(f"{main} vs {target}")
        safe_savefig(os.path.join(CHART_FOLDER, "regression.png"))
    except Exception as e:
        print("regression error:", e)

    # histogram
    try:
        plt.figure(figsize=(5, 4))
        sns.histplot(df[target], kde=True, bins=10)
        plt.xlabel(target)
        plt.ylabel("Count")
        plt.title(f"{target} distribution")
        safe_savefig(os.path.join(CHART_FOLDER, "hist.png"))
    except Exception as e:
        print("hist error:", e)


def train_models(df: pd.DataFrame):
    """Train models and store best model on disk."""
    global best_model_name, features_global, scores_global, feature_importances_global, target_col_global

    target = detect_target(df)
    target_col_global = target
    features = choose_features(df, target)
    features_global = features

    if not features:
        raise ValueError("No features to train – only target found")

    X = df[features].values
    y = df[target].values

    if len(y) < 5:
        raise ValueError("Need at least 5 rows to train models")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lin = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=160, random_state=42).fit(X_train, y_train)

    s_lin = lin.score(X_test, y_test)
    s_rf = rf.score(X_test, y_test)

    scores_global = {"Linear Regression": s_lin, "Random Forest": s_rf}
    best_model_name_local = max(scores_global, key=scores_global.get)
    best_model_name = best_model_name_local
    best_model = lin if best_model_name_local == "Linear Regression" else rf

    # Feature importance
    if isinstance(best_model, RandomForestRegressor):
        feature_importances_global = dict(
            zip(features, best_model.feature_importances_)
        )
    else:
        feature_importances_global = None

    os.makedirs(MODEL_FOLDER, exist_ok=True)
    joblib.dump(
        {"model": best_model, "target": target, "features": features},
        os.path.join(MODEL_FOLDER, "best.pkl"),
    )

    # small classifier for confusion matrix
    try:
        grades = np.array([grade_of(v) for v in y])
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, grades)
        labels = ["A", "B", "C", "D", "F"]
        cm = confusion_matrix(grades, clf.predict(X), labels=labels)
        plt.figure(figsize=(4.5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="magma",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Grade confusion matrix")
        safe_savefig(os.path.join(CHART_FOLDER, "confusion.png"))
    except Exception as e:
        print("confusion matrix error:", e)

    generate_charts(df, target, features)
    return scores_global, target


def dataset_summary(df: pd.DataFrame):
    rows, cols = df.shape
    numeric = df.select_dtypes(include=[np.number])
    summaries = []
    for c in numeric.columns:
        summaries.append(
            {
                "column": c,
                "mean": float(numeric[c].mean()),
                "min": float(numeric[c].min()),
                "max": float(numeric[c].max()),
            }
        )
    return {"rows": rows, "cols": cols, "numeric_summary": summaries}


# ================================================================
# PLAGIARISM HELPERS
# ================================================================

def read_submission_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".py", ".c", ".cpp", ".java", ".txt"]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""
    return ""


def compute_plagiarism_for_assignment(assign_id: int):
    prefix = f"ass{assign_id}_"
    files = [f for f in os.listdir(SUBMISSION_FOLDER) if f.startswith(prefix)]

    texts = []
    names = []
    for f in files:
        full = os.path.join(SUBMISSION_FOLDER, f)
        t = read_submission_text(full)
        if t.strip():
            texts.append(t)
            names.append(f)

    if len(texts) < 2:
        return []

    vec = TfidfVectorizer().fit(texts)
    tfidf = vec.transform(texts)
    sim = cosine_similarity(tfidf)

    n = len(names)
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            val = float(sim[i, j])
            if val >= 0.8:
                results.append(
                    {
                        "file1": names[i],
                        "file2": names[j],
                        "similarity": round(val * 100, 2),
                    }
                )
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results


# ================================================================
# LEADERBOARD HELPER
# ================================================================

def badge_for_percent(p: float) -> str:
    if p >= 90:
        return "Cyber Legend 🏆"
    if p >= 80:
        return "Neon Star ✨"
    if p >= 65:
        return "Consistent Coder 💻"
    if p >= 50:
        return "On Track 📘"
    return "Needs Support ❤️"


def build_leaderboard_for_teacher(teacher_username: str):
    grades = load_grades()
    if not grades:
        return []

    stats = {}
    for g in grades:
        if g.get("graded_by") != teacher_username:
            continue
        filename = g["filename"]
        parts = filename.split("_")
        if len(parts) >= 3 and parts[0].startswith("ass"):
            student = parts[1]
        else:
            student = "unknown"
        marks = float(g["marks"])
        out_of = float(g["out_of"])
        info = stats.setdefault(student, {"marks_sum": 0.0, "out_of_sum": 0.0, "count": 0})
        info["marks_sum"] += marks
        info["out_of_sum"] += out_of
        info["count"] += 1

    board = []
    for student, info in stats.items():
        if info["out_of_sum"] > 0:
            percent = 100.0 * info["marks_sum"] / info["out_of_sum"]
        else:
            percent = 0.0
        board.append(
            {
                "username": student,
                "avg_percent": round(percent, 2),
                "submissions": info["count"],
                "badge": badge_for_percent(percent),
            }
        )

    board.sort(key=lambda x: x["avg_percent"], reverse=True)
    for i, row in enumerate(board, start=1):
        row["rank"] = i
    return board


# ================================================================
# ONLINE IDE – PYTHON
# ================================================================

def run_python_code(code: str):
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return {"stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Execution timeout (5s)"}
    except Exception as e:
        return {"stdout": "", "stderr": str(e)}


# ================================================================
# ROUTES – BASIC
# ================================================================

@app.route("/")
def home():
    # a small "upcoming assignment" banner
    assignments = load_assignments()
    upcoming = None
    today = date.today()
    for a in assignments:
        if a.get("due_date"):
            try:
                d = datetime.strptime(a["due_date"], "%Y-%m-%d").date()
                left = (d - today).days
                if 0 <= left <= 3:
                    upcoming = {
                        "title": a["title"],
                        "due_date": a["due_date"],
                        "days_left": left,
                    }
                    break
            except Exception:
                continue
    return render_template("home.html", upcoming=upcoming)


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        row = get_user(username, password)
        if row:
            session["username"] = row["username"]
            session["role"] = row["role"]
            return redirect(url_for("home"))
        else:
            error = "Invalid username or password"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ================================================================
# ROUTES – PREDICTION (SEPARATE FROM PERFORMANCE)
# ================================================================

@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    """
    GET  -> show prediction form (predict.html).
    POST -> use uploaded model if exists, otherwise rule-based.
    NO performance charts here.
    """
    global target_col_global, features_global, best_model_name, scores_global

    if request.method == "POST":
        try:
            attendance = float(request.form.get("attendance", 0))
            study = float(request.form.get("study_hours", 0))
            assignments_val = float(request.form.get("assignments", 0))
            internal = float(request.form.get("internal_marks", 0))
            prev_sem = float(request.form.get("prev_percent", 0))

            # keep inputs so we can re-fill the form
            form_values = {
                "attendance": attendance,
                "study_hours": study,
                "assignments": assignments_val,
                "internal_marks": internal,
                "prev_percent": prev_sem,
            }

            input_features = {
                "attendance": attendance,
                "study_hours": study,
                "assignments": assignments_val,
                "internal_marks": internal,
                "prev_percent": prev_sem,
                "prev_sem": prev_sem,
            }

            pred_marks = None
            model_info_path = os.path.join(MODEL_FOLDER, "best.pkl")

            if os.path.exists(model_info_path):
                data = joblib.load(model_info_path)
                model = data["model"]
                model_feats = data["features"]
                row = [float(input_features.get(f, 0)) for f in model_feats]
                X = np.array([row])
                pred_marks = float(model.predict(X)[0])
                target_name = data.get("target", "Final Marks")
            else:
                # simple rule-based fallback
                base = 0.4 * prev_sem + 0.25 * internal + 0.2 * assignments_val + 0.15 * attendance
                pred_marks = min(max(base, 0), 100)
                target_name = target_col_global or "Final Marks"

            g = grade_of(pred_marks)

            tips = []
            if attendance < 75:
                tips.append("Increase attendance to at least 75–80%.")
            if study < 2:
                tips.append("Study at least 2–3 focused hours daily.")
            if assignments_val < 80:
                tips.append("Complete 80–90% of assignments on time.")
            if internal < prev_sem - 5:
                tips.append("Internal marks are below previous %, revise before tests.")
            if pred_marks < 60:
                tips.append("Practice previous year question papers weekly.")
            if not tips:
                tips.append("You are doing well. Maintain consistency and revision.")

            model_score = None
            if best_model_name and scores_global:
                model_score = round(scores_global.get(best_model_name, 0) * 100, 2)

            return render_template(
                "predict.html",
                submitted=True,
                form_values=form_values,
                prediction=round(pred_marks, 2),
                grade=g,
                target_name=target_name,
                tips=" ".join(tips),
                model_name=best_model_name,
                model_score=model_score,
            )
        except Exception as e:
            print("predict error:", e)
            return render_template(
                "predict.html",
                submitted=False,
                error="Could not run prediction. Please check your inputs.",
            )

    # GET
    return render_template("predict.html", submitted=False)


# ================================================================
# ROUTES – IDE
# ================================================================

@app.route("/ide")
@login_required()
def ide_page():
    return render_template("ide.html")


@app.route("/api/run_python", methods=["POST"])
@login_required()
def api_run_python():
    data = request.get_json() or {}
    code = data.get("code", "")
    if not code.strip():
        return jsonify({"stdout": "", "stderr": "No code provided"}), 400
    result = run_python_code(code)
    return jsonify(result)


# ================================================================
# ROUTES – DATASET UPLOAD + PERFORMANCE JSON
# ================================================================

@app.route("/upload")
@login_required(role="teacher")
def upload_page():
    return render_template("upload.html")


@app.route("/upload_dataset", methods=["POST"])
@login_required(role="teacher")
def upload_dataset():
    global current_df

    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file field"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".csv", ".xlsx", ".xls"]:
            return jsonify(
                {"success": False, "error": "Only CSV or Excel files allowed"}
            ), 400

        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(filepath)

        if ext == ".csv":
            current_df = pd.read_csv(filepath)
        else:
            current_df = pd.read_excel(filepath)

        if current_df is None or current_df.empty:
            return jsonify({"success": False, "error": "Uploaded file is empty"}), 400

        scores, target = train_models(current_df)
        summary = dataset_summary(current_df)

        return jsonify(
            {
                "success": True,
                "scores": scores,
                "target": target,
                "summary": summary,
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/performance")
@login_required(role="teacher")
def performance_page():
    global current_df, scores_global, best_model_name
    if current_df is None:
        return render_template(
            "performance.html",
            no_dataset=True,
            scores=None,
            best_model=None,
        )
    return render_template(
        "performance.html",
        no_dataset=False,
        scores=scores_global,
        best_model=best_model_name,
    )


@app.route("/api/performance/data")
@login_required(role="teacher")
def performance_data():
    global current_df, feature_importances_global, target_col_global

    if current_df is None:
        return jsonify({"error": "Dataset not uploaded"}), 400

    try:
        target = target_col_global or detect_target(current_df)
        y = current_df[target].dropna().values.astype(float)

        # grade distribution
        grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        for v in y:
            grade_counts[grade_of(v)] += 1

        # feature importance
        importance = feature_importances_global or {}

        # regression scatter points with first feature
        first_feat = features_global[0] if features_global else None
        scatter_points = []
        if first_feat and first_feat in current_df.columns:
            for xv, yv in zip(
                current_df[first_feat].values.tolist(),
                current_df[target].values.tolist(),
            ):
                scatter_points.append({"x": float(xv), "y": float(yv)})

        return jsonify(
            {
                "grades": grade_counts,
                "importance": importance,
                "target": target,
                "scatter_feature": first_feat,
                "scatter_points": scatter_points,
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ================================================================
# ROUTES – STUDENT DASHBOARD
# ================================================================

@app.route("/student/dashboard")
@login_required(role="student")
def student_dashboard():
    username = session.get("username")
    assignments = load_assignments()
    grades = load_grades()

    items = []
    total_marks = 0.0
    total_out = 0.0

    for a in assignments:
        assign_id = a["id"]
        prefix = f"ass{assign_id}_{username}_"
        submitted_files = [
            f for f in os.listdir(SUBMISSION_FOLDER) if f.startswith(prefix)
        ]
        is_sub = len(submitted_files) > 0
        grade_entries = [g for g in grades if g["filename"] in submitted_files]
        mark_text = "-"
        if grade_entries:
            g = grade_entries[0]
            mark_text = f"{g['marks']}/{g['out_of']}"
            total_marks += float(g["marks"])
            total_out += float(g["out_of"])

        items.append(
            {
                "assignment": a,
                "submitted": is_sub,
                "files": submitted_files,
                "marks": mark_text,
            }
        )

    avg_percent = None
    if total_out > 0:
        avg_percent = round(100 * total_marks / total_out, 2)

    return render_template("student_dashboard.html", items=items, avg_percent=avg_percent)


# ================================================================
# ROUTES – TEACHER ASSIGNMENTS
# ================================================================

@app.route("/teacher/assignments", methods=["GET", "POST"])
@login_required(role="teacher")
def teacher_assignments():
    username = session.get("username")
    assignments = load_assignments()
    my_assignments = [a for a in assignments if a.get("created_by") == username]

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        desc = request.form.get("description", "").strip()
        due_date = request.form.get("due_date", "").strip()
        file = request.files.get("file")

        if not title or not file:
            return render_template(
                "assignments_teacher.html",
                error="Title and a file are required.",
                assignments=my_assignments,
            )

        ext = os.path.splitext(file.filename)[1]
        new_id = (assignments[-1]["id"] + 1) if assignments else 1
        fname = f"assignment_{new_id}{ext}"
        fpath = os.path.join(ASSIGNMENT_FOLDER, fname)
        file.save(fpath)

        item = {
            "id": new_id,
            "title": title,
            "description": desc,
            "due_date": due_date,
            "file": fname,
            "created_by": username,
        }
        assignments.append(item)
        save_assignments(assignments)

        my_assignments = [a for a in assignments if a.get("created_by") == username]

        return render_template(
            "assignments_teacher.html",
            success="Assignment created.",
            assignments=my_assignments,
        )

    # GET
    return render_template("assignments_teacher.html", assignments=my_assignments)


@app.route("/teacher/assignments/<int:assign_id>/edit", methods=["GET", "POST"])
@login_required(role="teacher")
def edit_assignment(assign_id):
    username = session.get("username")
    assignments = load_assignments()
    assignment = next((a for a in assignments if a["id"] == assign_id and a["created_by"] == username), None)
    if not assignment:
        abort(404)

    if request.method == "POST":
        assignment["title"] = request.form.get("title", "").strip() or assignment["title"]
        assignment["description"] = request.form.get("description", "").strip()
        assignment["due_date"] = request.form.get("due_date", "").strip()
        save_assignments(assignments)
        return redirect(url_for("teacher_assignments"))

    return render_template("edit_assignment.html", assignment=assignment)


@app.route("/teacher/assignments/<int:assign_id>/delete", methods=["POST"])
@login_required(role="teacher")
def delete_assignment(assign_id):
    username = session.get("username")
    assignments = load_assignments()
    new_list = []
    deleted_file = None
    for a in assignments:
        if a["id"] == assign_id and a.get("created_by") == username:
            deleted_file = a.get("file")
            continue
        new_list.append(a)

    save_assignments(new_list)

    if deleted_file:
        try:
            os.remove(os.path.join(ASSIGNMENT_FOLDER, deleted_file))
        except FileNotFoundError:
            pass

    return redirect(url_for("teacher_assignments"))


@app.route("/assignments/file/<path:filename>")
@login_required()
def download_assignment_file(filename):
    return send_from_directory(ASSIGNMENT_FOLDER, filename, as_attachment=True)


# ================================================================
# ROUTES – STUDENT ASSIGNMENTS (SUBMISSION)
# ================================================================

@app.route("/student/assignments", methods=["GET", "POST"])
@login_required(role="student")
def student_assignments():
    username = session.get("username")
    assignments = load_assignments()
    error = None
    success = None

    if request.method == "POST":
        assign_id = request.form.get("assignment_id")
        if not assign_id:
            error = "Select an assignment."
        else:
            try:
                assign_id = int(assign_id)
            except ValueError:
                assign_id = None

        files = request.files.getlist("submission_files")

        if assign_id is None or not files:
            error = error or "Select an assignment and upload at least one file."
        else:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            for f in files:
                if not f.filename:
                    continue
                safe_name = secure_filename(f.filename)
                store_name = f"ass{assign_id}_{username}_{ts}_{safe_name}"
                path = os.path.join(SUBMISSION_FOLDER, store_name)
                f.save(path)
            if not error:
                success = "Submission uploaded."

    return render_template(
        "assignments_student.html",
        assignments=assignments,
        error=error,
        success=success,
    )


# ================================================================
# ROUTES – VIEW SUBMISSIONS / GRADING / ZIP / DOWNLOAD
# ================================================================

@app.route("/teacher/assignments/<int:assign_id>/submissions")
@login_required(role="teacher")
def view_submissions(assign_id):
    username = session.get("username")
    assignments = load_assignments()
    assignment = next((a for a in assignments if a["id"] == assign_id and a["created_by"] == username), None)
    if not assignment:
        abort(404)

    prefix = f"ass{assign_id}_"
    files = [f for f in os.listdir(SUBMISSION_FOLDER) if f.startswith(prefix)]
    grades = load_grades()

    submissions = []
    total_marks = 0.0
    total_out = 0.0
    for fname in files:
        grade = get_grade_for_file(fname)
        mark_info = None
        if grade:
            mark_info = grade
            total_marks += float(grade["marks"])
            total_out += float(grade["out_of"])

        # split name parts once here
        parts = fname.split("_")

        # detect late
        is_late = False
        if assignment.get("due_date"):
            try:
                due = datetime.strptime(assignment["due_date"], "%Y-%m-%d")
                if len(parts) >= 3:
                    ts_str = parts[2]
                    ts = datetime.strptime(ts_str, "%Y%m%d-%H%M%S")
                    is_late = ts > due
            except Exception:
                pass

        submitted_at_str = parts[2] if len(parts) >= 3 else "-"

        submissions.append(
            {
                "filename": fname,
                "grade": mark_info,
                "is_late": is_late,
                "submitted_at": submitted_at_str,
            }
        )

    avg_percent = None
    if total_out > 0:
        avg_percent = round(100 * total_marks / total_out, 2)

    return render_template(
        "submissions_list.html",
        submissions=submissions,
        assign_id=assign_id,
        assignment=assignment,
        avg_percent=avg_percent,
        avg_marks=round(total_marks, 2) if total_out > 0 else None,
    )


@app.route("/submissions/file/<path:filename>")
@login_required()
def download_submission(filename):
    return send_from_directory(SUBMISSION_FOLDER, filename, as_attachment=True)


@app.route("/teacher/assignments/<int:assign_id>/submissions/zip")
@login_required(role="teacher")
def download_all_submissions(assign_id):
    prefix = f"ass{assign_id}_"
    files = [f for f in os.listdir(SUBMISSION_FOLDER) if f.startswith(prefix)]
    if not files:
        abort(404)

    zip_name = f"ass{assign_id}_submissions.zip"
    zip_path = os.path.join(DATA_FOLDER, zip_name)

    if os.path.exists(zip_path):
        os.remove(zip_path)

    shutil.make_archive(zip_path[:-4], "zip", SUBMISSION_FOLDER)
    return send_file(zip_path, as_attachment=True)


@app.route("/grade/<path:filename>", methods=["GET", "POST"])
@login_required(role="teacher")
def grade_submission(filename):
    grades = load_grades()
    grade = get_grade_for_file(filename)

    if request.method == "POST":
        try:
            marks = float(request.form.get("marks", 0))
            out_of = float(request.form.get("out_of", 100))
        except ValueError:
            marks = 0.0
            out_of = 100.0
        feedback = request.form.get("feedback", "")

        # remove old entry if present
        grades = [g for g in grades if g.get("filename") != filename]
        grades.append(
            {
                "filename": filename,
                "marks": marks,
                "out_of": out_of,
                "feedback": feedback,
                "graded_by": session.get("username"),
            }
        )
        save_grades(grades)
        return redirect(request.referrer or url_for("teacher_assignments"))

    return render_template("grade_submission.html", filename=filename, grade=grade)


# ================================================================
# ROUTES – PLAGIARISM & LEADERBOARD
# ================================================================

@app.route("/teacher/assignments/<int:assign_id>/plagiarism")
@login_required(role="teacher")
def plagiarism_page(assign_id):
    username = session.get("username")
    assignments = load_assignments()
    assignment = next((a for a in assignments if a["id"] == assign_id and a["created_by"] == username), None)
    if not assignment:
        abort(404)

    pairs = compute_plagiarism_for_assignment(assign_id)
    return render_template(
        "plagiarism.html",
        assign_id=assign_id,
        assignment=assignment,
        pairs=pairs,
    )


@app.route("/leaderboard")
@login_required(role="teacher")
def leaderboard_page():
    board = build_leaderboard_for_teacher(session.get("username"))
    return render_template("leaderboard.html", leaderboard=board)


# ================================================================
# ERROR HANDLERS
# ================================================================

@app.errorhandler(404)
def not_found(e):
    return render_template("home.html", upcoming=None), 404


@app.errorhandler(500)
def server_error(e):
    # simple 500 page
    return "Internal Server Error", 500


# ================================================================
# MAIN ENTRYPOINT (LOCAL + RENDER)
# ================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug=True is fine locally; Render will set debug=False when using gunicorn
    app.run(host="0.0.0.0", port=port, debug=True)

