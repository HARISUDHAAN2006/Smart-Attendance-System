import os
import socket
import sqlite3
import datetime
import uuid
import pandas as pd
import qrcode
import io
import json
import numpy as np
from flask import Flask, render_template_string, request, redirect, url_for, send_file, session, flash, Response
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from contextlib import contextmanager
import math
import random

app = Flask(__name__)
app.secret_key = "attend_enterprise_2024"

DB_NAME = "attendance_enterprise.db"
ATTENDANCE_THRESHOLD = 75.0
TOTAL_SESSIONS_ESTIMATE = 40

SUBJECTS = [
    "Data Structures and Algorithms",
    "Machine Learning",
    "Business Intelligence and Data Visualization",
    "Artificial Intelligence / Cyber Ethics",
    "Database Management Systems"
]

def get_ip():
    if os.environ.get('ATTEND_HOST'):
        return os.environ.get('ATTEND_HOST')
    for target in [('8.8.8.8', 80), ('1.1.1.1', 80), ('10.255.255.255', 1)]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            s.connect(target)
            ip = s.getsockname()[0]
            s.close()
            if not ip.startswith('127.'):
                return ip
        except:
            pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if not ip.startswith('127.'):
            return ip
    except:
        pass
    return '127.0.0.1'

LOCAL_IP = get_ip()
PORT = 5000
BASE_URL = f"http://{LOCAL_IP}:{PORT}"

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS students
                     (student_id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT,
                      phone TEXT, department TEXT, year INTEGER DEFAULT 1)''')
        c.execute('''CREATE TABLE IF NOT EXISTS class_sessions
                     (token TEXT PRIMARY KEY, created_at DATETIME, expires_at DATETIME,
                      is_active INTEGER DEFAULT 0, subject TEXT, notes TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS attendance
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      student_id TEXT, session_token TEXT, timestamp DATETIME,
                      device_hash TEXT, is_anomaly INTEGER DEFAULT 0,
                      FOREIGN KEY(student_id) REFERENCES students(student_id) ON DELETE CASCADE,
                      FOREIGN KEY(session_token) REFERENCES class_sessions(token) ON DELETE CASCADE,
                      UNIQUE(student_id, session_token))''')
        c.execute('''CREATE TABLE IF NOT EXISTS app_settings
                     (key TEXT PRIMARY KEY, value TEXT)''')
        for k, v in [('threshold', '75'), ('total_sessions', '40'), ('org_name', 'IIT Delhi — CSE Dept')]:
            c.execute("INSERT OR IGNORE INTO app_settings (key, value) VALUES (?, ?)", (k, v))
        conn.commit()

def seed_dummy_data():
    """Seed rich Indian dummy data for demo purposes."""
    with get_db() as conn:
        existing = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        if existing > 0:
            return

        students = [
            ("STU2024001", "Aarav Sharma",       "aarav.sharma@iitd.ac.in",       "+91 9811234501", "Computer Science", 3),
            ("STU2024002", "Priya Patel",         "priya.patel@iitd.ac.in",        "+91 9811234502", "Computer Science", 3),
            ("STU2024003", "Rohan Mehta",         "rohan.mehta@iitd.ac.in",        "+91 9811234503", "Computer Science", 3),
            ("STU2024004", "Ananya Krishnan",     "ananya.k@iitd.ac.in",           "+91 9811234504", "Data Science",     2),
            ("STU2024005", "Vikram Nair",         "vikram.nair@iitd.ac.in",        "+91 9811234505", "Computer Science", 3),
            ("STU2024006", "Sneha Reddy",         "sneha.reddy@iitd.ac.in",        "+91 9811234506", "Data Science",     2),
            ("STU2024007", "Arjun Singh",         "arjun.singh@iitd.ac.in",        "+91 9811234507", "Computer Science", 4),
            ("STU2024008", "Kavya Iyer",          "kavya.iyer@iitd.ac.in",         "+91 9811234508", "AI & ML",          2),
            ("STU2024009", "Rahul Gupta",         "rahul.gupta@iitd.ac.in",        "+91 9811234509", "Computer Science", 3),
            ("STU2024010", "Divya Joshi",         "divya.joshi@iitd.ac.in",        "+91 9811234510", "Data Science",     3),
            ("STU2024011", "Aditya Kumar",        "aditya.kumar@iitd.ac.in",       "+91 9811234511", "AI & ML",          2),
            ("STU2024012", "Ishita Banerjee",     "ishita.b@iitd.ac.in",           "+91 9811234512", "Computer Science", 4),
            ("STU2024013", "Siddharth Rao",       "siddharth.rao@iitd.ac.in",      "+91 9811234513", "Computer Science", 3),
            ("STU2024014", "Pooja Verma",         "pooja.verma@iitd.ac.in",        "+91 9811234514", "Data Science",     2),
            ("STU2024015", "Karthik Subramanian", "karthik.s@iitd.ac.in",          "+91 9811234515", "AI & ML",          3),
            ("STU2024016", "Neha Agarwal",        "neha.agarwal@iitd.ac.in",       "+91 9811234516", "Computer Science", 2),
            ("STU2024017", "Manish Tiwari",       "manish.tiwari@iitd.ac.in",      "+91 9811234517", "Computer Science", 3),
            ("STU2024018", "Riya Desai",          "riya.desai@iitd.ac.in",         "+91 9811234518", "AI & ML",          2),
        ]
        for s in students:
            try:
                conn.execute("INSERT INTO students (student_id,name,email,phone,department,year) VALUES (?,?,?,?,?,?)", s)
            except:
                pass

        # Create 18 past sessions over last 30 days
        tokens = []
        base_date = datetime.datetime.now() - datetime.timedelta(days=30)
        for i, subj in enumerate(SUBJECTS * 4):
            if i >= 18: break
            t = str(uuid.uuid4())
            tokens.append((t, subj))
            dt = base_date + datetime.timedelta(days=i*1.6, hours=random.choice([9,10,11,14,15]))
            conn.execute("INSERT INTO class_sessions (token,created_at,is_active,subject,notes) VALUES (?,?,0,?,?)",
                         (t, dt.strftime("%Y-%m-%d %H:%M:%S"), subj, "Regular class"))

        conn.commit()

        # Seed attendance — realistic patterns: some students attend often, some rarely
        attendance_rates = {
            "STU2024001": 0.94, "STU2024002": 0.88, "STU2024003": 0.50,
            "STU2024004": 0.97, "STU2024005": 0.60, "STU2024006": 0.83,
            "STU2024007": 0.45, "STU2024008": 0.91, "STU2024009": 0.70,
            "STU2024010": 0.78, "STU2024011": 0.55, "STU2024012": 0.99,
            "STU2024013": 0.67, "STU2024014": 0.40, "STU2024015": 0.85,
            "STU2024016": 0.92, "STU2024017": 0.62, "STU2024018": 0.76,
        }

        base_date_d = base_date
        for idx, (token, subj) in enumerate(tokens):
            sess_dt = base_date_d + datetime.timedelta(days=idx * 1.6, hours=10)
            for sid, rate in attendance_rates.items():
                if random.random() < rate:
                    jitter = random.randint(30, 900)
                    att_time = sess_dt + datetime.timedelta(seconds=jitter)
                    fp = f"Mozilla/5.0|1440x900|en-IN_{sid}"
                    try:
                        conn.execute(
                            "INSERT INTO attendance (student_id,session_token,timestamp,device_hash,is_anomaly) VALUES (?,?,?,?,?)",
                            (sid, token, att_time.strftime("%Y-%m-%d %H:%M:%S"), fp, 0)
                        )
                    except:
                        pass
        conn.commit()

def get_setting(key, default=''):
    with get_db() as conn:
        row = conn.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
        return row['value'] if row else default

# ── ML & ANALYTICS ─────────────────────────────────────────────────────────
def calculate_stats(student_id):
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM class_sessions").fetchone()[0]
        if total == 0: return None
        present = conn.execute("SELECT COUNT(*) FROM attendance WHERE student_id=?", (student_id,)).fetchone()[0]
        threshold = float(get_setting('threshold', '75'))
        total_est = int(get_setting('total_sessions', '40'))
        pct = (present / total) * 100
        projected_pct = (present / total) * 100
        risk = "CRITICAL" if projected_pct < threshold else "SAFE"
        sessions_needed = max(0, int((threshold / 100 * total_est) - present))
        return {
            "percentage": round(pct, 1),
            "present": present,
            "total": total,
            "projected_pct": round(projected_pct, 1),
            "risk_level": risk,
            "sessions_needed": sessions_needed,
        }

def get_advanced_ml_insights():
    with sqlite3.connect(DB_NAME) as conn:
        students = pd.read_sql_query("SELECT student_id, name, department FROM students", conn)
        sessions_df = pd.read_sql_query("SELECT token, created_at FROM class_sessions", conn)
        att_df = pd.read_sql_query("SELECT student_id, session_token, timestamp, is_anomaly, device_hash FROM attendance", conn)

    if students.empty or sessions_df.empty:
        return []

    total = len(sessions_df)
    threshold = float(get_setting('threshold', '75'))
    total_est = int(get_setting('total_sessions', '40'))
    records = []

    for _, s in students.iterrows():
        sid = s['student_id']
        att = att_df[att_df['student_id'] == sid]
        count = len(att)
        anomalies = int(att['is_anomaly'].sum()) if not att.empty else 0
        pct = round((count / total * 100) if total > 0 else 0, 1)

        recency = 0
        if not att.empty and not sessions_df.empty:
            try:
                att_tokens = set(att['session_token'].tolist())
                recent = sessions_df.tail(5)['token'].tolist()
                recency = round(sum(1 for t in recent if t in att_tokens) / min(5, total) * 100, 1)
            except: recency = pct

        trend = 'stable'
        if count >= 3 and not sessions_df.empty:
            try:
                att_tokens = set(att['session_token'].tolist())
                y = np.array([1 if t in att_tokens else 0 for t in sessions_df['token'].tolist()])
                X = np.arange(len(y)).reshape(-1, 1)
                slope = LinearRegression().fit(X, y).coef_[0]
                trend = 'improving' if slope > 0.01 else ('declining' if slope < -0.01 else 'stable')
            except: pass

        risk_score = min(100, round(
            (100 - pct) * 0.5 + min(anomalies * 10, 30) + (100 - recency) * 0.2, 1))

        records.append({
            'student_id': sid, 'name': s['name'],
            'department': s['department'] or '',
            'attendance_pct': pct, 'anomaly_count': anomalies,
            'recency_score': recency, 'risk_score': risk_score,
            'sessions_present': count, 'trend': trend,
            'sessions_to_safe': max(0, int((threshold / 100 * total_est) - count)),
        })

    if len(records) >= 3:
        try:
            df = pd.DataFrame(records)
            feats = df[['attendance_pct', 'recency_score', 'anomaly_count']].fillna(0)
            scaled = StandardScaler().fit_transform(feats)
            labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)
            cluster_att = {i: feats.iloc[labels == i]['attendance_pct'].mean() for i in range(3)}
            sorted_c = sorted(cluster_att, key=cluster_att.get, reverse=True)
            cmap = {sorted_c[0]: 'Elite', sorted_c[1]: 'Average', sorted_c[2]: 'At-Risk'}
            for i, r in enumerate(records):
                r['cluster'] = cmap[labels[i]]
        except:
            for r in records: r['cluster'] = 'Unknown'
    else:
        for r in records: r['cluster'] = 'Unknown'

    return records

def detect_anomalies():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            df = pd.read_sql_query("""
                SELECT a.*, s.created_at as session_start
                FROM attendance a JOIN class_sessions s ON a.session_token = s.token
            """, conn)
            if len(df) < 5: return
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['session_start'] = pd.to_datetime(df['session_start'])
            df['sec_from_start'] = (df['timestamp'] - df['session_start']).dt.total_seconds().fillna(0)
            df['device_count'] = df.groupby('device_hash')['id'].transform('count')
            X = df[['sec_from_start', 'device_count']].fillna(0)
            preds = IsolationForest(contamination=0.1, random_state=42).fit_predict(X)
            df['is_anomaly'] = [1 if x == -1 else 0 for x in preds]
            for _, row in df.iterrows():
                conn.execute("UPDATE attendance SET is_anomaly=? WHERE id=?", (int(row['is_anomaly']), int(row['id'])))
            conn.commit()
    except Exception as e:
        print(f"ML anomaly error: {e}")

def get_trends():
    with get_db() as conn:
        daily = conn.execute("""SELECT DATE(timestamp) d, COUNT(*) c FROM attendance
                               GROUP BY DATE(timestamp) ORDER BY d DESC LIMIT 14""").fetchall()
        by_subject = conn.execute("""SELECT cs.subject, COUNT(a.id) c FROM class_sessions cs
                                     LEFT JOIN attendance a ON cs.token=a.session_token
                                     GROUP BY cs.subject""").fetchall()
        hourly = conn.execute("""SELECT strftime('%H',timestamp) h, COUNT(*) c FROM attendance
                                  GROUP BY h ORDER BY h""").fetchall()
        weekday = conn.execute("""SELECT strftime('%w',timestamp) w, COUNT(*) c FROM attendance
                                   GROUP BY w ORDER BY w""").fetchall()
    wmap = {0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'}
    return {
        'dates': [r['d'] for r in reversed(list(daily))],
        'counts': [r['c'] for r in reversed(list(daily))],
        'subject_labels': [r['subject'] for r in by_subject],
        'subject_counts': [r['c'] for r in by_subject],
        'hours': [f"{int(r['h']):02d}:00" for r in hourly],
        'hour_counts': [r['c'] for r in hourly],
        'weekdays': [wmap.get(int(r['w']), '?') for r in weekday],
        'weekday_counts': [r['c'] for r in weekday],
    }

# ─────────────────────────────────────────────────────────────────────────────
# SHARED PAGE LAYOUT  (enhanced)
# ─────────────────────────────────────────────────────────────────────────────
def page(content, title="Dashboard", active="dashboard"):
    org = get_setting('org_name', 'My Institution')
    nav_items = [
        ("dashboard",  "/admin",           "M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6", "Dashboard"),
        ("analytics",  "/analytics",       "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z", "Analytics"),
        ("students",   "/manage_students", "M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z", "Students"),
        ("sessions",   "/manage_sessions", "M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z", "Sessions"),
        ("reports",    "/reports",         "M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z", "Reports"),
        ("import",     "/bulk_import",     "M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12", "Import"),
        ("settings",   "/settings",        "M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z M15 12a3 3 0 11-6 0 3 3 0 016 0z", "Settings"),
    ]

    nav_html = ""
    for key, href, icon, label in nav_items:
        is_active = key == active
        nav_html += f"""
        <a href="{href}" class="nav-link {'nav-active' if is_active else ''}">
            <span class="nav-icon-wrap {'nav-icon-active' if is_active else ''}">
                <svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.9" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="{icon}"/>
                </svg>
            </span>
            <span class="nav-label">{label}</span>
            {'<span class="nav-pip"></span>' if is_active else ''}
        </a>"""

    return render_template_string("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>""" + title + """ — AttendIQ</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
/* ══ DESIGN TOKENS ══════════════════════════════════════════════════════════ */
:root {
    --bg:        #F0F2F8;
    --bg2:       #E8EBF4;
    --surface:   #FFFFFF;
    --surface2:  #F7F9FC;
    --glass:     rgba(255,255,255,0.72);
    --glass-b:   rgba(255,255,255,0.45);
    --border:    #DDE2EF;
    --border2:   #EEF1F8;
    --text:      #0D1117;
    --text2:     #4A5568;
    --muted:     #8A94A6;
    --blue:      #2F6FED;
    --blue2:     #1A4FBF;
    --blue-lt:   #EFF4FF;
    --blue-mid:  #C7D9FC;
    --indigo:    #4F46E5;
    --green:     #0EA86D;
    --green-lt:  #EAFAF3;
    --red:       #E53935;
    --red-lt:    #FFF0F0;
    --amber:     #F59E0B;
    --amber-lt:  #FFFBEB;
    --violet:    #7C3AED;
    --pink:      #EC4899;
    --sidebar:   256px;
    --radius:    14px;
    --shadow-sm: 0 1px 4px rgba(0,0,0,.06), 0 0 0 1px rgba(0,0,0,.04);
    --shadow:    0 4px 20px rgba(0,0,0,.08), 0 1px 4px rgba(0,0,0,.05);
    --shadow-lg: 0 12px 40px rgba(0,0,0,.12), 0 2px 8px rgba(0,0,0,.06);
    --glow-blue: 0 0 24px rgba(47,111,237,.18);
    --glow-green:0 0 24px rgba(14,168,109,.15);
}

/* ── BASE ── */
*, *::before, *::after { box-sizing: border-box; }
body {
    font-family: 'DM Sans', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0; font-size: 14px; line-height: 1.55;
    -webkit-font-smoothing: antialiased;
    background-image:
        radial-gradient(ellipse 80% 60% at 20% -10%, rgba(47,111,237,.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(79,70,229,.06) 0%, transparent 55%);
    background-attachment: fixed;
}
h1,h2,h3,h4,h5 { font-family: 'Syne', sans-serif; }
.mono { font-family: 'JetBrains Mono', monospace; }

/* ── SIDEBAR ── */
#sidebar {
    position: fixed; left: 0; top: 0; bottom: 0;
    width: var(--sidebar);
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
    z-index: 50; overflow-y: auto; overflow-x: hidden;
}
#sidebar::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 200px;
    background: linear-gradient(160deg, rgba(47,111,237,.06) 0%, transparent 100%);
    pointer-events: none;
}

.sidebar-brand { padding: 22px 20px 18px; border-bottom: 1px solid var(--border2); }
.brand-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, var(--blue), var(--indigo));
    border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 12px rgba(47,111,237,.35);
    flex-shrink: 0;
}
.sidebar-section { padding: 18px 14px 6px; }
.sidebar-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; color: var(--muted);
    padding: 0 8px; margin-bottom: 6px; display: block;
}

/* Nav links */
.nav-link {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 10px; border-radius: 10px;
    text-decoration: none; font-size: 13.5px; font-weight: 500;
    color: var(--text2); margin-bottom: 2px;
    transition: all .22s cubic-bezier(.4,0,.2,1);
    position: relative; overflow: hidden;
}
.nav-link::before {
    content: '';
    position: absolute; inset: 0; border-radius: 10px;
    background: linear-gradient(90deg, var(--blue-lt), transparent);
    opacity: 0; transition: opacity .22s;
}
.nav-link:hover { color: var(--blue); background: var(--blue-lt); }
.nav-link:hover::before { opacity: 1; }
.nav-active { color: var(--blue) !important; background: var(--blue-lt) !important; font-weight: 600; }
.nav-active::before { opacity: 1; }
.nav-icon-wrap {
    width: 30px; height: 30px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    background: transparent; flex-shrink: 0;
    transition: background .22s, transform .22s;
}
.nav-link:hover .nav-icon-wrap { background: rgba(47,111,237,.1); transform: scale(1.05); }
.nav-icon-active { background: rgba(47,111,237,.14) !important; }
.nav-label { flex: 1; }
.nav-pip {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--blue); flex-shrink: 0;
    box-shadow: 0 0 6px rgba(47,111,237,.5);
    animation: pipPulse 3s ease infinite;
}
@keyframes pipPulse {
    0%,100% { opacity:1; } 50% { opacity:.5; }
}

/* ── MAIN ── */
#main { margin-left: var(--sidebar); min-height: 100vh; }
.topbar {
    background: rgba(255,255,255,.88);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    padding: 14px 32px;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 40;
}
.page-content { padding: 32px; }

/* ── GLASS CARDS ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    transition: box-shadow .25s ease, border-color .25s ease, transform .25s ease;
}
.card-glass {
    background: var(--glass);
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,.7);
}
.card:hover { box-shadow: var(--shadow); }
.card-hover:hover {
    box-shadow: 0 8px 32px rgba(47,111,237,.12);
    border-color: var(--blue-mid);
    transform: translateY(-3px);
}
.card-header {
    padding: 18px 22px;
    border-bottom: 1px solid var(--border2);
    display: flex; align-items: center; justify-content: space-between;
}

/* ── STAT CARDS ── */
.stat-card {
    padding: 22px; position: relative; overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute; right: -20px; bottom: -20px;
    width: 90px; height: 90px; border-radius: 50%;
    background: currentColor; opacity: .04;
    transition: transform .4s ease;
}
.stat-card:hover::after { transform: scale(1.5); }
.stat-icon {
    width: 46px; height: 46px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; position: relative;
}
.stat-icon::before {
    content: '';
    position: absolute; inset: 0; border-radius: 12px;
    background: inherit; opacity: .2; filter: blur(8px);
    transform: translateY(3px);
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem; font-weight: 800; line-height: 1.05;
    letter-spacing: -.03em;
}

/* ── BADGES ── */
.badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 9px; border-radius: 6px;
    font-size: 11px; font-weight: 600; letter-spacing: .015em;
    font-family: 'JetBrains Mono', monospace;
}
.badge-blue   { background: var(--blue-lt);  color: var(--blue);  border: 1px solid var(--blue-mid); }
.badge-green  { background: var(--green-lt); color: var(--green); border: 1px solid #A3E6CC; }
.badge-red    { background: var(--red-lt);   color: var(--red);   border: 1px solid #FFCDD2; }
.badge-amber  { background: var(--amber-lt); color: var(--amber); border: 1px solid #FDE68A; }
.badge-gray   { background: #F1F5F9; color: #475569; border: 1px solid #DDE2EF; }
.badge-violet { background: #F5F3FF; color: var(--violet); border: 1px solid #DDD6FE; }

/* ── BUTTONS ── */
.btn {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 9px 18px; border-radius: 9px;
    font-size: 13.5px; font-weight: 600; cursor: pointer;
    border: none; transition: all .2s cubic-bezier(.4,0,.2,1);
    text-decoration: none; font-family: 'DM Sans', sans-serif;
    position: relative; overflow: hidden;
}
.btn::after {
    content: '';
    position: absolute; inset: 0;
    background: rgba(255,255,255,0);
    transition: background .2s;
}
.btn:hover::after { background: rgba(255,255,255,.12); }
.btn-primary {
    background: linear-gradient(135deg, var(--blue), var(--indigo));
    color: white;
    box-shadow: 0 2px 8px rgba(47,111,237,.35), inset 0 1px 0 rgba(255,255,255,.15);
}
.btn-primary:hover {
    box-shadow: 0 6px 20px rgba(47,111,237,.42);
    transform: translateY(-1px);
}
.btn-primary:active { transform: translateY(0); }
.btn-secondary {
    background: var(--surface); color: var(--text);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
}
.btn-secondary:hover { background: var(--bg); border-color: #C0CBE0; }
.btn-danger {
    background: var(--red-lt); color: var(--red);
    border: 1px solid #FFCDD2;
}
.btn-danger:hover { background: var(--red); color: white; border-color: var(--red); }
.btn-sm { padding: 6px 13px; font-size: 12.5px; }

/* ── FORMS ── */
.form-group { margin-bottom: 18px; }
.form-label {
    display: block; font-size: 12.5px; font-weight: 600;
    color: var(--text); margin-bottom: 6px; letter-spacing: .01em;
}
.form-input, .form-select, .form-textarea {
    width: 100%; padding: 9px 13px;
    background: var(--surface); border: 1.5px solid var(--border);
    border-radius: 9px; font-size: 13.5px; color: var(--text);
    font-family: 'DM Sans', sans-serif;
    transition: border-color .18s, box-shadow .18s; outline: none;
}
.form-input:focus, .form-select:focus, .form-textarea:focus {
    border-color: var(--blue);
    box-shadow: 0 0 0 3.5px rgba(47,111,237,.13);
}
.form-hint { font-size: 12px; color: var(--muted); margin-top: 5px; }

/* ── TABLES ── */
.data-table { width: 100%; border-collapse: collapse; }
.data-table thead th {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .08em; color: var(--muted);
    padding: 11px 16px; text-align: left;
    border-bottom: 1px solid var(--border);
    background: var(--surface2);
    white-space: nowrap;
}
.data-table tbody td {
    padding: 13px 16px; font-size: 13.5px;
    border-bottom: 1px solid var(--border2);
    color: var(--text2); vertical-align: middle;
}
.data-table tbody tr { transition: background .15s; }
.data-table tbody tr:hover { background: #F0F4FF; }
.data-table tbody tr:last-child td { border-bottom: none; }

/* ── PROGRESS ── */
.prog-bar {
    height: 6px; background: var(--border);
    border-radius: 3px; overflow: hidden;
}
.prog-fill {
    height: 100%; border-radius: 3px;
    transition: width 1.4s cubic-bezier(.16,1,.3,1);
    position: relative; overflow: hidden;
}
.prog-fill::after {
    content: '';
    position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.4), transparent);
    animation: shimmer 2.2s infinite;
}
@keyframes shimmer { to { left: 200%; } }

/* ── LIVE DOT ── */
.live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green); display: inline-block;
    animation: livePulse 2.5s ease infinite;
}
@keyframes livePulse {
    0%  { box-shadow: 0 0 0 0 rgba(14,168,109,.5); }
    60% { box-shadow: 0 0 0 8px rgba(14,168,109,0); }
    100%{ box-shadow: 0 0 0 0 rgba(14,168,109,0); }
}

/* ── ANIMATIONS ── */
@keyframes fadeUp {
    from { opacity:0; transform:translateY(22px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeIn { from { opacity:0; } to { opacity:1; } }
@keyframes scaleIn {
    from { opacity:0; transform:scale(.94); }
    to   { opacity:1; transform:scale(1); }
}
@keyframes slideRight {
    from { opacity:0; transform:translateX(-16px); }
    to   { opacity:1; transform:translateX(0); }
}
.anim   { animation: fadeUp .55s cubic-bezier(.16,1,.3,1) both; }
.anim-s { animation: scaleIn .45s cubic-bezier(.16,1,.3,1) both; }
.anim-in{ animation: fadeIn .4s ease both; }
.d1  { animation-delay: 40ms; }
.d2  { animation-delay: 90ms; }
.d3  { animation-delay: 140ms; }
.d4  { animation-delay: 190ms; }
.d5  { animation-delay: 250ms; }
.d6  { animation-delay: 310ms; }
.d7  { animation-delay: 380ms; }

/* ── COUNTER ROLL ── */
.counter-roll {
    display: inline-block;
    transition: all .05s ease;
}

/* ── TOAST ── */
.toast {
    position: fixed; bottom: 28px; right: 28px; z-index: 1000;
    background: var(--text); color: white;
    border-radius: 12px; padding: 14px 22px;
    font-size: 13.5px; font-weight: 500;
    display: flex; align-items: center; gap: 10px;
    box-shadow: 0 12px 40px rgba(0,0,0,.25);
    animation: toastIn .45s cubic-bezier(.16,1,.3,1) both,
               toastOut .3s ease 3.8s both forwards;
    max-width: 360px; border: 1px solid rgba(255,255,255,.08);
}
@keyframes toastIn  { from { opacity:0; transform:translateY(20px) scale(.95); } to { opacity:1; transform:translateY(0) scale(1); } }
@keyframes toastOut { from { opacity:1; } to { opacity:0; transform:translateY(10px); } }

/* ── QR ── */
.qr-wrapper {
    background: white; border-radius: 14px;
    padding: 14px; display: inline-block;
    box-shadow: 0 8px 32px rgba(47,111,237,.2);
    animation: qrFloat 4s ease-in-out infinite;
}
@keyframes qrFloat {
    0%,100% { transform: translateY(0);   box-shadow: 0 8px 32px rgba(47,111,237,.2); }
    50%      { transform: translateY(-6px);box-shadow: 0 16px 40px rgba(47,111,237,.3); }
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #C8D0E0; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #A0AABC; }

/* ── SECTION HEADER ── */
.section-title { font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 700; color: var(--text); }
.section-sub   { font-size: 12.5px; color: var(--muted); margin-top: 2px; }

/* ── TREND ── */
.trend-up   { color: var(--green); font-weight: 600; }
.trend-down { color: var(--red); font-weight: 600; }
.trend-flat { color: var(--muted); }

/* ── CLUSTER ── */
.chip-elite   { background:#EAFAF3; color:#065F46; border:1px solid #A3E6CC; }
.chip-average { background:#FFFBEB; color:#78350F; border:1px solid #FDE68A; }
.chip-atrisk  { background:#FFF0F0; color:#991B1B; border:1px solid #FFCDD2; }

/* ── SEARCH ── */
.search-wrap { position: relative; }
.search-wrap svg { position: absolute; left: 10px; top: 50%; transform: translateY(-50%); pointer-events: none; }
.search-wrap input { padding-left: 34px; }

/* ── ROW LINK ── */
.row-link { cursor: pointer; }
.row-link:hover td { background: #EDF2FF !important; }

/* ── GRADIENT LINES ── */
.grad-line {
    height: 3px; border-radius: 2px;
    background: linear-gradient(90deg, var(--blue), var(--indigo), var(--violet));
}

/* ── AVATAR ── */
.avatar {
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 14px;
    flex-shrink: 0; position: relative;
}
.avatar::after {
    content: ''; position: absolute; inset: 0; border-radius: 10px;
    border: 1.5px solid rgba(255,255,255,.5);
}

/* ── HEATMAP ── */
.heatmap-cell {
    width: 14px; height: 14px; border-radius: 3px;
    display: inline-block; cursor: pointer;
    transition: transform .15s, filter .15s;
}
.heatmap-cell:hover { transform: scale(1.4); filter: brightness(1.15); z-index: 1; position: relative; }

/* ── SPARKLINE ── */
.spark { display: inline-flex; align-items: flex-end; gap: 2px; height: 24px; }
.spark-bar {
    width: 5px; border-radius: 2px;
    background: var(--blue-mid);
    transition: height .5s cubic-bezier(.16,1,.3,1), background .2s;
}
.spark-bar:hover { background: var(--blue); }

/* ── LEADERBOARD ── */
.lb-row {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 14px; border-radius: 10px;
    transition: background .18s; cursor: pointer; text-decoration: none;
}
.lb-row:hover { background: var(--blue-lt); }
.lb-rank {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 15px;
    width: 28px; text-align: center; flex-shrink: 0;
}

/* ── PARTICLE CANVAS ── */
#particles { position: fixed; inset: 0; pointer-events: none; z-index: 0; opacity: .4; }

/* ── MOBILE ── */
#mob-toggle { display: none; }
@media (max-width: 900px) {
    #sidebar { transform: translateX(-100%); transition: transform .3s ease; }
    #sidebar.open { transform: translateX(0); }
    #main { margin-left: 0; }
    #mob-toggle { display: flex; }
    .page-content { padding: 20px 16px; }
}
</style>
</head>
<body>

<!-- Particle background -->
<canvas id="particles"></canvas>

<!-- Sidebar -->
<aside id="sidebar">
    <div class="sidebar-brand">
        <div style="display:flex;align-items:center;gap:11px">
            <div class="brand-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
                    <circle cx="9" cy="7" r="4"/>
                    <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
                    <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
                </svg>
            </div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:16px;color:#0D1117;letter-spacing:-.02em">Attend<span style="color:var(--blue)">IQ</span></div>
                <div class="mono" style="font-size:9.5px;color:var(--muted);letter-spacing:.06em">ENTERPRISE v2</div>
            </div>
        </div>
    </div>

    <div class="sidebar-section">
        <span class="sidebar-label">Overview</span>
        """ + nav_html[:nav_html.index("Analytics") + 50] + """
    </div>
    <div class="sidebar-section">
        <span class="sidebar-label">Manage</span>
        """ + nav_html[nav_html.index("Students"):nav_html.index("Import") + 50] + """
    </div>
    <div class="sidebar-section">
        <span class="sidebar-label">System</span>
        """ + nav_html[nav_html.index("Settings"):] + """
    </div>

    <div style="margin-top:auto;padding:16px 14px;border-top:1px solid var(--border2)">
        <div style="background:linear-gradient(135deg,var(--blue-lt),#F0F4FF);border:1px solid var(--blue-mid);border-radius:11px;padding:13px 14px">
            <div style="display:flex;align-items:center;gap:7px;margin-bottom:5px">
                <span class="live-dot"></span>
                <span class="mono" style="font-size:9.5px;color:var(--blue);font-weight:600;letter-spacing:.08em">NODE ACTIVE</span>
            </div>
            <div class="mono" style="font-size:11px;color:var(--text2);font-weight:500">{{ ip }}</div>
            <div style="font-size:11px;color:var(--muted);margin-top:2px">Port {{ port }}</div>
        </div>
        <a href="/export_excel" class="btn btn-secondary btn-sm" style="width:100%;justify-content:center;margin-top:10px;display:flex">
            <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
            Export CSV
        </a>
    </div>
</aside>

<!-- Main -->
<div id="main">
    <header class="topbar">
        <div style="display:flex;align-items:center;gap:14px">
            <button id="mob-toggle" onclick="document.getElementById('sidebar').classList.toggle('open')"
                    style="background:none;border:1px solid var(--border);border-radius:8px;padding:6px 8px;cursor:pointer">
                <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" d="M4 6h16M4 12h16M4 18h16"/>
                </svg>
            </button>
            <div>
                <h1 style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;margin:0;letter-spacing:-.01em">""" + title + """</h1>
                <p style="font-size:12px;color:var(--muted);margin:0">""" + org + """</p>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:10px">
            <div style="font-size:12px;color:var(--muted);padding:6px 12px;background:var(--bg);border-radius:8px;border:1px solid var(--border)" class="mono" id="liveClock"></div>
            <a href="/register_student" class="btn btn-primary btn-sm">
                <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z"/></svg>
                Enroll Student
            </a>
        </div>
    </header>

    {% with messages = get_flashed_messages() %}
    {% if messages %}{% for msg in messages %}
    <div class="toast">
        <svg width="16" height="16" fill="none" stroke="#0EA86D" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>
        {{ msg }}
    </div>
    {% endfor %}{% endif %}
    {% endwith %}

    <div class="page-content">
        """ + content + """
    </div>
</div>

<script>
/* ── PARTICLE SYSTEM ── */
(function(){
    const canvas = document.getElementById('particles');
    const ctx = canvas.getContext('2d');
    let W, H, dots=[];
    function resize(){ W=canvas.width=window.innerWidth; H=canvas.height=window.innerHeight; }
    resize(); window.addEventListener('resize', resize);
    for(let i=0;i<55;i++) dots.push({
        x: Math.random()*2000, y: Math.random()*1200,
        vx:(Math.random()-.5)*.22, vy:(Math.random()-.5)*.22,
        r: Math.random()*2+1,
        o: Math.random()*.35+.05
    });
    function draw(){
        ctx.clearRect(0,0,W,H);
        dots.forEach(d=>{
            d.x+=d.vx; d.y+=d.vy;
            if(d.x<0||d.x>W) d.vx*=-1;
            if(d.y<0||d.y>H) d.vy*=-1;
            ctx.beginPath();
            ctx.arc(d.x,d.y,d.r,0,Math.PI*2);
            ctx.fillStyle=`rgba(47,111,237,${d.o})`;
            ctx.fill();
        });
        // Connect nearby dots
        dots.forEach((a,i)=>{
            dots.slice(i+1).forEach(b=>{
                const dist=Math.hypot(a.x-b.x,a.y-b.y);
                if(dist<130){
                    ctx.beginPath();
                    ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y);
                    ctx.strokeStyle=`rgba(47,111,237,${.12*(1-dist/130)})`;
                    ctx.lineWidth=.8; ctx.stroke();
                }
            });
        });
        requestAnimationFrame(draw);
    }
    draw();
})();

/* ── LIVE CLOCK ── */
function updateClock(){
    const now = new Date();
    document.getElementById('liveClock').textContent =
        now.toLocaleTimeString('en-IN',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
}
setInterval(updateClock,1000); updateClock();

/* ── PROGRESS BARS ── */
document.querySelectorAll('[data-prog]').forEach(el=>{
    const w=el.getAttribute('data-prog');
    el.style.width='0%';
    requestAnimationFrame(()=>setTimeout(()=>el.style.width=w+'%',80));
});

/* ── ANIMATED COUNTERS ── */
document.querySelectorAll('[data-count]').forEach(el=>{
    const target=parseInt(el.getAttribute('data-count'));
    const duration=1200; const start=performance.now();
    function tick(now){
        const t=Math.min((now-start)/duration,1);
        const ease=1-Math.pow(1-t,4);
        el.textContent=Math.round(ease*target);
        if(t<1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
});

/* ── TABLE SEARCH ── */
const searchInput=document.getElementById('tableSearch');
if(searchInput){
    searchInput.addEventListener('input',function(){
        const q=this.value.toLowerCase();
        document.querySelectorAll('[data-searchrow]').forEach(row=>{
            row.style.display=row.textContent.toLowerCase().includes(q)?'':'none';
        });
    });
}

/* ── RIPPLE ── */
document.querySelectorAll('.btn').forEach(btn=>{
    btn.addEventListener('click',function(e){
        const r=document.createElement('span');
        const rect=this.getBoundingClientRect();
        const size=Math.max(rect.width,rect.height)*2;
        r.style.cssText=`position:absolute;border-radius:50%;transform:scale(0);animation:rippleAnim .5s linear;
            background:rgba(255,255,255,.25);width:${size}px;height:${size}px;
            left:${e.clientX-rect.left-size/2}px;top:${e.clientY-rect.top-size/2}px;pointer-events:none`;
        this.appendChild(r);
        setTimeout(()=>r.remove(),500);
    });
});
</script>
<style>
@keyframes rippleAnim{to{transform:scale(1);opacity:0}}
</style>
</body>
</html>""", ip=LOCAL_IP, port=PORT)


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/admin')
def admin_panel():
    with get_db() as conn:
        n_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        n_logs     = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
        n_anomaly  = conn.execute("SELECT COUNT(*) FROM attendance WHERE is_anomaly=1").fetchone()[0]
        n_sessions = conn.execute("SELECT COUNT(*) FROM class_sessions").fetchone()[0]
        active_sess= conn.execute("SELECT * FROM class_sessions WHERE is_active=1 ORDER BY created_at DESC LIMIT 1").fetchone()
        live_count = 0
        if active_sess:
            live_count = conn.execute("SELECT COUNT(*) FROM attendance WHERE session_token=?",
                                      (active_sess['token'],)).fetchone()[0]
        logs = conn.execute("""
            SELECT a.student_id, s.name, a.timestamp, a.is_anomaly, cs.subject
            FROM attendance a JOIN students s ON a.student_id=s.student_id
            JOIN class_sessions cs ON a.session_token=cs.token
            ORDER BY a.timestamp DESC LIMIT 10""").fetchall()
        all_students = conn.execute("SELECT student_id, name FROM students").fetchall()

    risk_list = []
    safe_list = []
    for s in all_students:
        st = calculate_stats(s['student_id'])
        if st and st['risk_level'] == 'CRITICAL':
            risk_list.append({'name': s['name'], 'id': s['student_id'], 'pct': st['percentage']})
        elif st:
            safe_list.append({'name': s['name'], 'id': s['student_id'], 'pct': st['percentage']})
    risk_list.sort(key=lambda x: x['pct'])
    top_students = sorted(safe_list, key=lambda x: -x['pct'])[:5]

    trends = get_trends()

    today_count = 0
    with get_db() as conn:
        today_count = conn.execute("SELECT COUNT(*) FROM attendance WHERE DATE(timestamp)=DATE('now')").fetchone()[0]

    avg_attendance = 0
    if all_students:
        vals = [calculate_stats(s['student_id'])['percentage'] for s in all_students if calculate_stats(s['student_id'])]
        avg_attendance = round(sum(vals)/len(vals), 1) if vals else 0

    # Heatmap: last 70 days
    with get_db() as conn:
        heatmap_raw = conn.execute("""
            SELECT DATE(timestamp) d, COUNT(*) c FROM attendance
            WHERE timestamp >= DATE('now','-70 days')
            GROUP BY DATE(timestamp)""").fetchall()
    heatmap_dict = {r['d']: r['c'] for r in heatmap_raw}
    heatmap_days = []
    base = datetime.date.today() - datetime.timedelta(days=69)
    max_c = max(heatmap_dict.values(), default=1)
    for i in range(70):
        d = (base + datetime.timedelta(days=i)).isoformat()
        c = heatmap_dict.get(d, 0)
        intensity = min(int(c / max_c * 4), 4) if c > 0 else 0
        colors = ['#EEF1F8','#C7D9FC','#93B8FA','#5C94F8','#2F6FED']
        heatmap_days.append((d, c, colors[intensity]))

    heatmap_cells = ''.join(
        f'<span class="heatmap-cell" style="background:{col}" title="{d}: {c} check-ins"></span>'
        for d, c, col in heatmap_days
    )

    qr_section = ""
    if active_sess:
        qr_section = f"""
        <div class="card d6 anim" style="padding:22px;background:linear-gradient(135deg,#EFF4FF 0%,#F0FDF4 100%);border-color:var(--blue-mid)">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
                <div style="display:flex;align-items:center;gap:8px">
                    <span class="live-dot"></span>
                    <span style="font-weight:700;font-size:13px;color:var(--green);font-family:'Syne',sans-serif">Session Live</span>
                </div>
                <a href="/end_session/{active_sess['token']}" class="btn btn-danger btn-sm">
                    <svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/></svg>
                    End
                </a>
            </div>
            <p style="font-weight:700;font-size:13.5px;margin-bottom:4px;color:var(--text)">{active_sess['subject']}</p>
            <p style="font-size:12px;color:var(--muted);margin-bottom:16px"><span style="color:var(--blue);font-weight:600">{live_count}</span> checked in</p>
            <div style="text-align:center">
                <div class="qr-wrapper">
                    <img src="/qr/{active_sess['token']}" width="155" height="155" style="display:block;border-radius:6px">
                </div>
                <a href="/?token={active_sess['token']}" target="_blank"
                   style="display:inline-block;margin-top:10px;font-size:12px;color:var(--blue);text-decoration:none;font-weight:600">
                    Open check-in →
                </a>
            </div>
        </div>"""

    # Leaderboard HTML
    lb_colors = ['#FFD700','#C0C8D0','#CD7F32','#2F6FED','#2F6FED']
    lb_emojis = ['🥇','🥈','🥉','4','5']
    leaderboard_html = ''.join([f"""
    <a href="/student/{s['id']}" class="lb-row">
        <div class="lb-rank" style="color:{lb_colors[i]}">{lb_emojis[i]}</div>
        <div class="avatar" style="background:linear-gradient(135deg,{'#EFF4FF' if i<3 else '#F5F3FF'},{'#C7D9FC' if i<3 else '#DDD6FE'});color:{'#2F6FED' if i<3 else '#7C3AED'}">
            {s['name'][0]}
        </div>
        <div style="flex:1;min-width:0">
            <div style="font-weight:600;font-size:13px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{s['name']}</div>
            <div style="margin-top:4px">
                <div class="prog-bar" style="height:5px"><div class="prog-fill" data-prog="{s['pct']}" style="background:linear-gradient(90deg,var(--blue),var(--indigo));width:0%"></div></div>
            </div>
        </div>
        <span class="mono" style="font-size:13px;font-weight:700;color:var(--green);flex-shrink:0">{s['pct']}%</span>
    </a>""" for i, s in enumerate(top_students)])

    content = f"""
<!-- KPI Row -->
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:22px">
    <div class="card stat-card card-hover anim d1" style="color:var(--blue)">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:16px">
            <div class="stat-icon" style="background:linear-gradient(135deg,#EFF4FF,#C7D9FC)">
                <svg width="20" height="20" fill="none" stroke="var(--blue)" stroke-width="1.8" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z"/></svg>
            </div>
            <span class="badge badge-blue">Enrolled</span>
        </div>
        <div class="stat-value" data-count="{n_students}">0</div>
        <p class="mono" style="font-size:10.5px;color:var(--muted);margin-top:5px;text-transform:uppercase;letter-spacing:.07em">Total Students</p>
    </div>
    <div class="card stat-card card-hover anim d2" style="color:var(--green)">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:16px">
            <div class="stat-icon" style="background:linear-gradient(135deg,#EAFAF3,#A3E6CC)">
                <svg width="20" height="20" fill="none" stroke="var(--green)" stroke-width="1.8" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            </div>
            <span class="badge badge-green">Logged</span>
        </div>
        <div class="stat-value" data-count="{n_logs}">0</div>
        <p class="mono" style="font-size:10.5px;color:var(--muted);margin-top:5px;text-transform:uppercase;letter-spacing:.07em">Total Check-ins</p>
    </div>
    <div class="card stat-card card-hover anim d3" style="color:var(--red)">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:16px">
            <div class="stat-icon" style="background:linear-gradient(135deg,#FFF0F0,#FFCDD2)">
                <svg width="20" height="20" fill="none" stroke="var(--red)" stroke-width="1.8" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
            </div>
            <span class="badge badge-red">ML Flagged</span>
        </div>
        <div class="stat-value" data-count="{n_anomaly}">0</div>
        <p class="mono" style="font-size:10.5px;color:var(--muted);margin-top:5px;text-transform:uppercase;letter-spacing:.07em">Anomalies</p>
    </div>
    <div class="card stat-card card-hover anim d4" style="color:var(--amber)">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:16px">
            <div class="stat-icon" style="background:linear-gradient(135deg,#FFFBEB,#FDE68A)">
                <svg width="20" height="20" fill="none" stroke="var(--amber)" stroke-width="1.8" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
            </div>
            <span class="badge badge-amber">{today_count} today</span>
        </div>
        <div class="stat-value" data-count="{n_sessions}">0</div>
        <p class="mono" style="font-size:10.5px;color:var(--muted);margin-top:5px;text-transform:uppercase;letter-spacing:.07em">Total Sessions</p>
    </div>
</div>

<!-- Avg attendance banner -->
<div class="card anim d2" style="padding:16px 24px;margin-bottom:22px;background:linear-gradient(135deg,#0D1117,#1A2540);border:none;overflow:hidden;position:relative">
    <div style="position:absolute;right:-30px;top:-40px;width:200px;height:200px;border-radius:50%;background:rgba(47,111,237,.12)"></div>
    <div style="position:absolute;right:80px;bottom:-60px;width:150px;height:150px;border-radius:50%;background:rgba(79,70,229,.1)"></div>
    <div style="display:flex;align-items:center;justify-content:space-between;position:relative">
        <div style="display:flex;align-items:center;gap:20px">
            <div>
                <div style="font-size:11px;color:rgba(255,255,255,.5);font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.1em;margin-bottom:3px">Class Avg Attendance</div>
                <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;color:white;letter-spacing:-.03em">{avg_attendance}<span style="font-size:20px;color:rgba(255,255,255,.6)">%</span></div>
            </div>
            <div style="width:1px;height:48px;background:rgba(255,255,255,.1)"></div>
            <div>
                <div style="font-size:11px;color:rgba(255,255,255,.5);font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.1em;margin-bottom:3px">At Risk</div>
                <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;color:#FF6B6B;letter-spacing:-.03em">{len(risk_list)}</div>
            </div>
            <div style="width:1px;height:48px;background:rgba(255,255,255,.1)"></div>
            <div>
                <div style="font-size:11px;color:rgba(255,255,255,.5);font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.1em;margin-bottom:3px">Safe</div>
                <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;color:#34EEB0;letter-spacing:-.03em">{len(safe_list)}</div>
            </div>
        </div>
        <div style="text-align:right">
            <div style="font-size:11px;color:rgba(255,255,255,.4);margin-bottom:6px">Threshold</div>
            <div class="mono" style="font-size:22px;font-weight:700;color:rgba(255,255,255,.7)">{get_setting('threshold','75')}%</div>
        </div>
    </div>
</div>

<!-- Middle Row -->
<div style="display:grid;grid-template-columns:2fr 1fr;gap:16px;margin-bottom:22px">
    <!-- Trend Chart -->
    <div class="card anim d2">
        <div class="card-header">
            <div><div class="section-title">Attendance Trend</div><div class="section-sub">Daily check-in volume — last 14 days</div></div>
            <div style="display:flex;align-items:center;gap:8px">
                <span class="live-dot" style="width:7px;height:7px"></span>
                <span class="badge badge-green">Live</span>
            </div>
        </div>
        <div style="padding:20px 22px 16px">
            <div style="height:230px;position:relative"><canvas id="trendChart"></canvas></div>
        </div>
    </div>

    <!-- Launch + QR -->
    <div style="display:flex;flex-direction:column;gap:14px">
        <div class="card anim d3" style="padding:22px">
            <div class="section-title" style="margin-bottom:3px">Launch Session</div>
            <div class="section-sub" style="margin-bottom:16px">Generate QR attendance gate</div>
            <form action="/start_session" method="POST">
                <div class="form-group">
                    <label class="form-label">Subject</label>
                    <select name="subject" class="form-select">
                        {''.join([f'<option>{s}</option>' for s in SUBJECTS])}
                    </select>
                </div>
                <button type="submit" class="btn btn-primary" style="width:100%;justify-content:center">
                    <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 4v1m6.364 1.636l-.707.707M20 12h-1M17.657 17.657l-.707-.707M12 20v-1m-5.657-1.636l.707-.707M4 12h1m2.343-5.657l.707.707"/></svg>
                    Generate QR Gate
                </button>
            </form>
        </div>
        {qr_section}
    </div>
</div>

<!-- Bottom Row -->
<div style="display:grid;grid-template-columns:3fr 2fr;gap:16px;margin-bottom:22px">
    <!-- Live Telemetry -->
    <div class="card anim d4">
        <div class="card-header">
            <div style="display:flex;align-items:center;gap:8px">
                <span class="live-dot"></span>
                <div class="section-title">Live Telemetry</div>
            </div>
            <span class="badge badge-gray">Last 10 scans</span>
        </div>
        <div style="overflow:hidden">
            <table class="data-table">
                <thead><tr><th>Student</th><th>Subject</th><th>Time</th><th>Status</th></tr></thead>
                <tbody>
                    {''.join([f"""
                    <tr class="row-link" onclick="window.location='/student/{log[0]}'">
                        <td>
                            <div style="display:flex;align-items:center;gap:10px">
                                <div class="avatar" style="background:{'linear-gradient(135deg,#FFF0F0,#FFCDD2)' if log[3] else 'linear-gradient(135deg,#EFF4FF,#C7D9FC)'};color:{'var(--red)' if log[3] else 'var(--blue)'}">
                                    {log[1][0].upper()}
                                </div>
                                <div>
                                    <div style="font-weight:600;font-size:13px;color:var(--text)">{log[1]}</div>
                                    <div class="mono" style="font-size:10px;color:var(--muted)">{log[0]}</div>
                                </div>
                            </div>
                        </td>
                        <td style="max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text2);font-size:13px">{log[4]}</td>
                        <td class="mono" style="font-size:11.5px;color:var(--muted)">{str(log[2])[:16]}</td>
                        <td>{'<span class="badge badge-red">⚠ Anomaly</span>' if log[3] else '<span class="badge badge-green">✓ Valid</span>'}</td>
                    </tr>""" for log in logs]) or '<tr><td colspan="4" style="text-align:center;padding:40px;color:var(--muted)">No records yet</td></tr>'}
                </tbody>
            </table>
        </div>
    </div>

    <!-- At-Risk Panel -->
    <div class="card anim d5">
        <div class="card-header">
            <div>
                <div class="section-title">At-Risk Students</div>
                <div class="section-sub">Below {ATTENDANCE_THRESHOLD}% threshold</div>
            </div>
            <span class="badge badge-red">{len(risk_list)} flagged</span>
        </div>
        <div style="padding:10px;max-height:340px;overflow-y:auto">
            {''.join([f"""
            <a href="/student/{r['id']}" style="display:flex;align-items:center;gap:11px;padding:10px;border-radius:10px;text-decoration:none;transition:background .15s;margin-bottom:2px"
               onmouseover="this.style.background='#FFF0F0'" onmouseout="this.style.background=''">
                <div class="avatar" style="background:linear-gradient(135deg,#FFF0F0,#FFCDD2);color:var(--red)">{r['name'][0].upper()}</div>
                <div style="flex:1;min-width:0">
                    <div style="font-weight:600;font-size:13px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{r['name']}</div>
                    <div style="margin-top:5px">
                        <div class="prog-bar"><div class="prog-fill" data-prog="{r['pct']}" style="background:linear-gradient(90deg,var(--red),#FF6B6B);width:0%"></div></div>
                    </div>
                </div>
                <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:16px;color:var(--red);flex-shrink:0">{r['pct']}%</div>
            </a>""" for r in risk_list]) or '<div style="text-align:center;padding:36px 20px;color:var(--muted)"><div style="font-size:30px;margin-bottom:8px">✓</div><div style="font-weight:600;font-size:13px;color:var(--green)">All students on track!</div></div>'}
        </div>
    </div>
</div>

<!-- Heatmap + Leaderboard row -->
<div style="display:grid;grid-template-columns:3fr 2fr;gap:16px">
    <!-- Heatmap -->
    <div class="card anim d5">
        <div class="card-header">
            <div><div class="section-title">Attendance Heatmap</div><div class="section-sub">Check-in density — last 70 days</div></div>
            <div style="display:flex;align-items:center;gap:5px">
                <span style="font-size:11px;color:var(--muted)">Less</span>
                {''.join([f'<span style="width:12px;height:12px;border-radius:3px;background:{c};display:inline-block"></span>' for c in ['#EEF1F8','#C7D9FC','#93B8FA','#5C94F8','#2F6FED']])}
                <span style="font-size:11px;color:var(--muted)">More</span>
            </div>
        </div>
        <div style="padding:20px;display:flex;flex-wrap:wrap;gap:3px">
            {heatmap_cells}
        </div>
    </div>

    <!-- Leaderboard -->
    <div class="card anim d6">
        <div class="card-header">
            <div><div class="section-title">Top Performers</div><div class="section-sub">Highest attendance rate</div></div>
            <span class="badge badge-violet">🏆 Ranked</span>
        </div>
        <div style="padding:10px">
            {leaderboard_html or '<div style="text-align:center;padding:30px;color:var(--muted)">No data yet</div>'}
        </div>
    </div>
</div>

<script>
document.addEventListener('DOMContentLoaded',()=>{{
    const ctx=document.getElementById('trendChart').getContext('2d');
    const grad=ctx.createLinearGradient(0,0,0,230);
    grad.addColorStop(0,'rgba(47,111,237,0.18)');
    grad.addColorStop(0.6,'rgba(79,70,229,0.06)');
    grad.addColorStop(1,'rgba(47,111,237,0)');
    new Chart(ctx,{{
        type:'line',
        data:{{
            labels:{json.dumps(trends['dates'])},
            datasets:[{{
                label:'Check-ins',
                data:{json.dumps(trends['counts'])},
                borderColor:'#2F6FED', borderWidth:2.5,
                backgroundColor:grad, fill:true, tension:0.44,
                pointBackgroundColor:'#fff',
                pointBorderColor:'#2F6FED',
                pointBorderWidth:2.5, pointRadius:4, pointHoverRadius:7,
                pointHoverBackgroundColor:'#2F6FED',
            }}]
        }},
        options:{{
            responsive:true, maintainAspectRatio:false,
            plugins:{{
                legend:{{display:false}},
                tooltip:{{ backgroundColor:'#0D1117', titleColor:'#EEF1F8', bodyColor:'#8A94A6',
                    padding:13, cornerRadius:10, borderColor:'#1E2A3A', borderWidth:1,
                    titleFont:{{family:'Syne',weight:'700'}}, bodyFont:{{family:'DM Sans'}} }}
            }},
            scales:{{
                y:{{ beginAtZero:true, grid:{{color:'rgba(0,0,0,.04)'}},
                     ticks:{{color:'#8A94A6',font:{{size:11,family:'JetBrains Mono'}}}} }},
                x:{{ grid:{{display:false}},
                     ticks:{{color:'#8A94A6',font:{{size:11,family:'JetBrains Mono'}}}} }}
            }},
            animation:{{ duration:1600, easing:'easeOutQuart' }}
        }}
    }});
}});
</script>
"""
    return page(content, "Dashboard", "dashboard")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/analytics')
def analytics_page():
    ml_data = get_advanced_ml_insights()
    trends  = get_trends()

    if not ml_data:
        content = """<div style="text-align:center;padding:80px 0">
            <div style="font-size:40px;margin-bottom:16px">📊</div>
            <h2 style="font-family:'Syne',sans-serif;font-size:20px;font-weight:700;margin-bottom:8px">No data yet</h2>
            <p style="color:var(--muted)">Add students and run sessions to generate analytics.</p>
            <a href="/admin" class="btn btn-primary" style="display:inline-flex;margin-top:20px">Back to Dashboard</a>
        </div>"""
        return page(content, "Analytics", "analytics")

    elite = sum(1 for r in ml_data if r['cluster'] == 'Elite')
    avg   = sum(1 for r in ml_data if r['cluster'] == 'Average')
    risk  = sum(1 for r in ml_data if r['cluster'] == 'At-Risk')
    names = [r['name'].split()[0] for r in ml_data]
    risk_scores = [r['risk_score'] for r in ml_data]
    colors = ['#0EA86D' if r['cluster']=='Elite' else '#F59E0B' if r['cluster']=='Average' else '#E53935' for r in ml_data]

    trend_html = {
        'improving':'<span class="trend-up">▲ Improving</span>',
        'declining': '<span class="trend-down">▼ Declining</span>',
        'stable':    '<span class="trend-flat">● Stable</span>'
    }

    content = f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:26px" class="anim">
    <div>
        <h1 style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;margin:0;letter-spacing:-.03em">ML Analytics</h1>
        <p style="color:var(--muted);font-size:13px;margin-top:3px">Isolation Forest · K-Means · Linear Regression</p>
    </div>
    <a href="/admin" class="btn btn-secondary btn-sm">← Dashboard</a>
</div>

<!-- Cluster Cards -->
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px">
    <div class="card anim d1" style="padding:22px;background:linear-gradient(135deg,#EAFAF3,#F0FFF9);border-color:#A3E6CC;overflow:hidden;position:relative">
        <div style="position:absolute;right:-15px;top:-15px;width:80px;height:80px;border-radius:50%;background:rgba(14,168,109,.08)"></div>
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
            <span class="badge chip-elite" style="font-size:11px">Elite Cluster</span>
            <div style="width:36px;height:36px;background:rgba(14,168,109,.12);border-radius:10px;display:flex;align-items:center;justify-content:center">
                <svg width="18" height="18" fill="none" stroke="#0EA86D" stroke-width="2.2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 3l14 9-14 9V3z"/></svg>
            </div>
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:44px;font-weight:800;color:#059669;line-height:1">{elite}</div>
        <div class="mono" style="font-size:10px;color:#065F46;margin-top:4px;letter-spacing:.08em">HIGH PERFORMERS</div>
    </div>
    <div class="card anim d2" style="padding:22px;background:linear-gradient(135deg,#FFFBEB,#FEFCE8);border-color:#FDE68A;overflow:hidden;position:relative">
        <div style="position:absolute;right:-15px;top:-15px;width:80px;height:80px;border-radius:50%;background:rgba(245,158,11,.06)"></div>
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
            <span class="badge chip-average">Average Cluster</span>
            <div style="width:36px;height:36px;background:rgba(245,158,11,.12);border-radius:10px;display:flex;align-items:center;justify-content:center">
                <svg width="18" height="18" fill="none" stroke="#F59E0B" stroke-width="2.2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M20 12H4"/></svg>
            </div>
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:44px;font-weight:800;color:#D97706;line-height:1">{avg}</div>
        <div class="mono" style="font-size:10px;color:#78350F;margin-top:4px;letter-spacing:.08em">MID-TIER</div>
    </div>
    <div class="card anim d3" style="padding:22px;background:linear-gradient(135deg,#FFF0F0,#FFF5F5);border-color:#FFCDD2;overflow:hidden;position:relative">
        <div style="position:absolute;right:-15px;top:-15px;width:80px;height:80px;border-radius:50%;background:rgba(229,57,53,.07)"></div>
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
            <span class="badge chip-atrisk">At-Risk Cluster</span>
            <div style="width:36px;height:36px;background:rgba(229,57,53,.12);border-radius:10px;display:flex;align-items:center;justify-content:center">
                <svg width="18" height="18" fill="none" stroke="#E53935" stroke-width="2.2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
            </div>
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:44px;font-weight:800;color:#E53935;line-height:1">{risk}</div>
        <div class="mono" style="font-size:10px;color:#991B1B;margin-top:4px;letter-spacing:.08em">NEEDS ATTENTION</div>
    </div>
</div>

<!-- Charts -->
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px">
    <div class="card anim d2">
        <div class="card-header"><div class="section-title">Risk Score Distribution</div><span class="badge badge-violet">ML Composite</span></div>
        <div style="padding:20px;height:260px"><canvas id="riskChart"></canvas></div>
    </div>
    <div class="card anim d3">
        <div class="card-header"><div class="section-title">Subject Breakdown</div><span class="badge badge-gray">Check-ins per subject</span></div>
        <div style="padding:20px;height:260px"><canvas id="subjectChart"></canvas></div>
    </div>
    <div class="card anim d4">
        <div class="card-header"><div class="section-title">Hourly Pattern</div><span class="badge badge-blue">Peak check-in times</span></div>
        <div style="padding:20px;height:200px"><canvas id="hourChart"></canvas></div>
    </div>
    <div class="card anim d5">
        <div class="card-header"><div class="section-title">Weekly Pattern</div><span class="badge badge-green">Day-of-week attendance</span></div>
        <div style="padding:20px;height:200px"><canvas id="weekChart"></canvas></div>
    </div>
</div>

<!-- ML Table -->
<div class="card anim d5">
    <div class="card-header">
        <div>
            <div class="section-title">Student ML Analysis</div>
            <div class="section-sub">{len(ml_data)} students · K-Means clustering applied</div>
        </div>
        <div class="search-wrap">
            <svg width="14" height="14" fill="none" stroke="#94A3B8" stroke-width="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path stroke-linecap="round" d="m21 21-4.35-4.35"/></svg>
            <input id="tableSearch" class="form-input" placeholder="Search students…" style="width:200px;padding:7px 10px 7px 30px">
        </div>
    </div>
    <div style="overflow-x:auto">
        <table class="data-table">
            <thead><tr>
                <th>Student</th><th>Attendance</th><th>Risk Score</th><th>Cluster</th>
                <th>Trend</th><th>Recency</th><th>Anomalies</th><th>Sessions Needed</th>
            </tr></thead>
            <tbody>
                {''.join([f"""
                <tr class="row-link" data-searchrow onclick="window.location='/student/{r['student_id']}'">
                    <td>
                        <div style="display:flex;align-items:center;gap:10px">
                            <div class="avatar" style="background:{'linear-gradient(135deg,#EAFAF3,#A3E6CC)' if r['cluster']=='Elite' else 'linear-gradient(135deg,#FFFBEB,#FDE68A)' if r['cluster']=='Average' else 'linear-gradient(135deg,#FFF0F0,#FFCDD2)'};color:{'#059669' if r['cluster']=='Elite' else '#D97706' if r['cluster']=='Average' else '#E53935'}">
                                {r['name'][0].upper()}
                            </div>
                            <div>
                                <div style="font-weight:600;font-size:13px">{r['name']}</div>
                                <div class="mono" style="font-size:10px;color:var(--muted)">{r['student_id']}</div>
                            </div>
                        </div>
                    </td>
                    <td>
                        <div style="display:flex;align-items:center;gap:8px">
                            <div style="width:64px"><div class="prog-bar"><div class="prog-fill" data-prog="{r['attendance_pct']}"
                                style="background:{'linear-gradient(90deg,var(--green),#34EEB0)' if r['attendance_pct']>=75 else 'linear-gradient(90deg,var(--red),#FF6B6B)'};width:0%"></div></div></div>
                            <span class="mono" style="font-size:12px;font-weight:600;color:{'var(--green)' if r['attendance_pct']>=75 else 'var(--red)'}">{r['attendance_pct']}%</span>
                        </div>
                    </td>
                    <td>
                        <div style="display:flex;align-items:center;gap:8px">
                            <div style="width:48px"><div class="prog-bar"><div class="prog-fill" data-prog="{r['risk_score']}"
                                style="background:{'linear-gradient(90deg,var(--red),#FF6B6B)' if r['risk_score']>60 else 'linear-gradient(90deg,var(--amber),#FCD34D)' if r['risk_score']>30 else 'linear-gradient(90deg,var(--green),#34EEB0)'};width:0%"></div></div></div>
                            <span class="mono" style="font-size:12px;font-weight:700;color:{'var(--red)' if r['risk_score']>60 else 'var(--amber)' if r['risk_score']>30 else 'var(--green)'}">{r['risk_score']}</span>
                        </div>
                    </td>
                    <td><span class="badge {'chip-elite' if r['cluster']=='Elite' else 'chip-average' if r['cluster']=='Average' else 'chip-atrisk'}">{r['cluster']}</span></td>
                    <td style="font-size:13px">{trend_html.get(r['trend'],'—')}</td>
                    <td class="mono" style="font-size:12px">{r['recency_score']}%</td>
                    <td><span class="badge {'badge-red' if r['anomaly_count']>0 else 'badge-green'}">{r['anomaly_count']}</span></td>
                    <td>
                        {'<span class="badge badge-red">⚠ ' + str(r["sessions_to_safe"]) + ' more</span>' if r['sessions_to_safe'] > 0 else '<span class="badge badge-green">✓ Safe</span>'}
                    </td>
                </tr>""" for r in ml_data])}
            </tbody>
        </table>
    </div>
</div>

<script>
document.addEventListener('DOMContentLoaded',()=>{{
    const base = {{
        responsive:true, maintainAspectRatio:false,
        plugins:{{ legend:{{display:false}},
            tooltip:{{ backgroundColor:'#0D1117', titleColor:'#EEF1F8', bodyColor:'#8A94A6',
                padding:13, cornerRadius:10, borderColor:'#1E2A3A', borderWidth:1 }} }},
        scales:{{
            y:{{ grid:{{color:'rgba(0,0,0,.04)'}}, ticks:{{color:'#8A94A6',font:{{size:10,family:'JetBrains Mono'}}}} }},
            x:{{ grid:{{display:false}}, ticks:{{color:'#8A94A6',font:{{size:10,family:'JetBrains Mono'}}}} }}
        }},
        animation:{{ duration:1400, easing:'easeOutQuart' }}
    }};
    new Chart(document.getElementById('riskChart').getContext('2d'),{{
        type:'bar', data:{{ labels:{json.dumps(names)},
            datasets:[{{ data:{json.dumps(risk_scores)}, backgroundColor:{json.dumps(colors)},
                borderRadius:8, borderSkipped:false }}] }},
        options:{{...base}}
    }});
    new Chart(document.getElementById('subjectChart').getContext('2d'),{{
        type:'doughnut',
        data:{{ labels:{json.dumps(trends['subject_labels'])},
            datasets:[{{ data:{json.dumps(trends['subject_counts'])},
                backgroundColor:['#2F6FED','#0EA86D','#F59E0B','#E53935','#7C3AED'],
                borderColor:'#fff', borderWidth:3, hoverOffset:8 }}] }},
        options:{{ responsive:true, maintainAspectRatio:false, cutout:'65%',
            plugins:{{ legend:{{ display:true, position:'right',
                labels:{{ color:'#4A5568', font:{{size:11,family:'DM Sans'}}, boxWidth:10, padding:14 }} }},
                tooltip:{{ backgroundColor:'#0D1117', titleColor:'#EEF1F8', bodyColor:'#8A94A6',padding:12,cornerRadius:10 }} }},
            animation:{{ duration:1400, easing:'easeOutQuart' }} }}
    }});
    new Chart(document.getElementById('hourChart').getContext('2d'),{{
        type:'bar', data:{{ labels:{json.dumps(trends['hours'])},
            datasets:[{{ data:{json.dumps(trends['hour_counts'])},
                backgroundColor:'rgba(47,111,237,0.72)',
                borderRadius:6, borderSkipped:false }}] }},
        options:{{...base}}
    }});
    new Chart(document.getElementById('weekChart').getContext('2d'),{{
        type:'bar', data:{{ labels:{json.dumps(trends['weekdays'])},
            datasets:[{{ data:{json.dumps(trends['weekday_counts'])},
                backgroundColor:'rgba(14,168,109,0.72)',
                borderRadius:6, borderSkipped:false }}] }},
        options:{{...base}}
    }});
}});
</script>
"""
    return page(content, "Analytics", "analytics")


# ─────────────────────────────────────────────────────────────────────────────
# STUDENT PROFILE
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/student/<sid>')
def view_student(sid):
    with get_db() as conn:
        student = conn.execute("SELECT * FROM students WHERE student_id=?", (sid,)).fetchone()
        if not student: return redirect('/manage_students')
        logs = conn.execute("""SELECT a.timestamp, cs.subject, a.is_anomaly
            FROM attendance a JOIN class_sessions cs ON a.session_token=cs.token
            WHERE a.student_id=? ORDER BY a.timestamp DESC LIMIT 20""", (sid,)).fetchall()
        by_subject = conn.execute("""
            SELECT cs.subject, COUNT(*) c FROM attendance a
            JOIN class_sessions cs ON a.session_token=cs.token
            WHERE a.student_id=? GROUP BY cs.subject""", (sid,)).fetchall()

    stats = calculate_stats(sid)
    pct   = stats['percentage'] if stats else 0
    risk  = stats['risk_level'] if stats else 'N/A'
    color = '#0EA86D' if risk == 'SAFE' else '#E53935'
    circ  = 2 * 3.14159 * 45
    dash  = pct / 100 * circ

    subject_rows = ''.join([f"""
    <div style="display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--border2)">
        <span style="font-size:13px;color:var(--text2);flex:1;padding-right:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{s['subject']}</span>
        <div style="display:flex;align-items:center;gap:10px;flex-shrink:0">
            <div style="width:70px"><div class="prog-bar"><div class="prog-fill" data-prog="{min(100,s['c']*20)}" style="background:linear-gradient(90deg,var(--blue),var(--indigo));width:0%"></div></div></div>
            <span class="mono" style="font-size:12px;font-weight:600;color:var(--text)">{s['c']}</span>
        </div>
    </div>""" for s in by_subject]) or '<p style="color:var(--muted);font-size:13px;padding:16px 0">No subject data yet.</p>'

    content = f"""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:28px" class="anim">
    <a href="/manage_students" style="color:var(--muted);text-decoration:none;font-size:13px;display:flex;align-items:center;gap:5px">
        <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M15 18l-6-6 6-6"/></svg>
        Students
    </a>
    <span style="color:var(--border)">/</span>
    <div style="width:52px;height:52px;background:linear-gradient(135deg,#EFF4FF,#C7D9FC);border-radius:14px;
         display:flex;align-items:center;justify-content:center;font-family:'Syne',sans-serif;font-weight:800;font-size:22px;color:var(--blue);
         box-shadow:0 4px 14px rgba(47,111,237,.2)">
        {student['name'][0].upper()}
    </div>
    <div>
        <h1 style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;margin:0;letter-spacing:-.02em">{student['name']}</h1>
        <div style="display:flex;align-items:center;gap:8px;margin-top:5px;flex-wrap:wrap">
            <span class="badge badge-blue mono">{student['student_id']}</span>
            {f'<span class="badge badge-gray">{student["email"]}</span>' if student['email'] else ''}
            {f'<span class="badge badge-violet">{student["department"]}</span>' if student['department'] else ''}
            {f'<span class="badge badge-gray">Year {student["year"]}</span>' if student['year'] else ''}
        </div>
    </div>
</div>

<div style="display:grid;grid-template-columns:220px 1fr;gap:16px;margin-bottom:20px">
    <!-- Ring -->
    <div class="card anim d1" style="padding:28px;text-align:center">
        <div class="mono" style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:18px">Standing</div>
        <div style="position:relative;width:115px;height:115px;margin:0 auto 16px">
            <svg viewBox="0 0 100 100" width="115" height="115" style="transform:rotate(-90deg)">
                <circle cx="50" cy="50" r="45" fill="none" stroke="#EEF1F8" stroke-width="8"/>
                <circle cx="50" cy="50" r="45" fill="none" stroke="{color}" stroke-width="8"
                    stroke-linecap="round"
                    stroke-dasharray="{dash:.1f} {circ:.1f}"
                    style="transition:stroke-dasharray 1.6s cubic-bezier(.16,1,.3,1);
                    filter:drop-shadow(0 0 6px {'rgba(14,168,109,.3)' if risk=='SAFE' else 'rgba(229,57,53,.3)'})"/>
            </svg>
            <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center">
                <span style="font-family:'Syne',sans-serif;font-weight:800;font-size:22px;color:{color};line-height:1">{pct}%</span>
                <span style="font-size:10px;color:var(--muted);margin-top:2px">{stats['present'] if stats else 0}/{stats['total'] if stats else 0}</span>
            </div>
        </div>
        <span class="badge {'badge-green' if risk=='SAFE' else 'badge-red'}" style="font-size:12px">{risk}</span>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
        <div class="card anim d2" style="padding:20px;{'background:linear-gradient(135deg,#FFF0F0,#FFF5F5);border-color:#FFCDD2' if stats and stats['risk_level']=='CRITICAL' else 'background:linear-gradient(135deg,#EAFAF3,#F0FFF9);border-color:#A3E6CC'}">
            <div class="mono" style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px">AI Forecast</div>
            <div style="font-family:'Syne',sans-serif;font-size:42px;font-weight:800;color:{color};line-height:1">{stats['projected_pct'] if stats else '—'}<span style="font-size:22px">%</span></div>
            <p style="font-size:12px;color:var(--muted);margin-top:6px">Projected final rate</p>
            {f'<div style="margin-top:12px;padding:10px 12px;background:white;border-radius:9px;border:1px solid #FFCDD2;font-size:12.5px;font-weight:600;color:var(--red)">⚠ Need {stats["sessions_needed"]} more sessions</div>' if stats and stats['risk_level']=='CRITICAL' else '<div style="margin-top:12px;padding:10px 12px;background:white;border-radius:9px;border:1px solid #A3E6CC;font-size:12.5px;font-weight:600;color:var(--green)">✓ On track for semester</div>' if stats else ''}
        </div>
        <div class="card anim d3" style="padding:20px">
            <div class="mono" style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px">By Subject</div>
            {subject_rows}
        </div>
    </div>
</div>

<!-- Timeline -->
<div class="card anim d4">
    <div class="card-header">
        <div class="section-title">Attendance Timeline</div>
        <span class="badge badge-gray">Last 20 records</span>
    </div>
    <div style="overflow-x:auto">
        <table class="data-table">
            <thead><tr><th>Timestamp</th><th>Subject</th><th>Verification</th></tr></thead>
            <tbody>
                {''.join([f"""
                <tr>
                    <td class="mono" style="font-size:11.5px">{log['timestamp']}</td>
                    <td style="font-size:13px;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{log['subject']}</td>
                    <td>{'<span class="badge badge-red">⚠ Flagged</span>' if log['is_anomaly'] else '<span class="badge badge-green">✓ Valid</span>'}</td>
                </tr>""" for log in logs]) or '<tr><td colspan="3" style="text-align:center;padding:40px;color:var(--muted)">No records</td></tr>'}
            </tbody>
        </table>
    </div>
</div>
"""
    return page(content, student['name'], "students")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK-IN  (student-facing, mobile optimised)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def student_view():
    token = request.args.get('token', '').strip()
    with get_db() as conn:
        sess = conn.execute("SELECT * FROM class_sessions WHERE token=?", (token,)).fetchone()

    if sess and sess['is_active'] == 1:
        return render_template_string(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0">
<title>Check In — {sess['subject']}</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{
    font-family:'DM Sans',sans-serif;
    background:linear-gradient(135deg,#0D1117 0%,#1A2540 50%,#0D1117 100%);
    color:#0D1117;min-height:100vh;
    display:flex;align-items:center;justify-content:center;padding:20px;
    overflow:hidden;
}}
.bg-orb{{position:fixed;border-radius:50%;filter:blur(60px);pointer-events:none}}
.card{{
    background:rgba(255,255,255,.97);
    border-radius:24px;
    box-shadow:0 24px 80px rgba(0,0,0,.4),0 0 0 1px rgba(255,255,255,.12);
    padding:40px 36px;max-width:420px;width:100%;position:relative;overflow:hidden;
    animation:cardIn .7s cubic-bezier(.16,1,.3,1) both;
}}
@keyframes cardIn{{from{{opacity:0;transform:translateY(30px) scale(.96)}}to{{opacity:1;transform:translateY(0) scale(1)}}}}
.accent-bar{{
    position:absolute;top:0;left:0;right:0;height:4px;
    background:linear-gradient(90deg,#2F6FED,#4F46E5,#7C3AED);
}}
.subject{{font-family:'Syne',sans-serif;font-size:21px;font-weight:800;color:#0D1117;margin:14px 0 4px;letter-spacing:-.02em}}
.sub-label{{font-size:12px;color:#8A94A6;margin-bottom:28px}}
.live-badge{{
    display:inline-flex;align-items:center;gap:6px;
    background:linear-gradient(135deg,#EAFAF3,#F0FFF9);
    border:1px solid #A3E6CC;border-radius:20px;
    padding:5px 14px;font-size:12px;font-weight:600;color:#059669;margin-bottom:20px;
}}
.dot{{width:7px;height:7px;background:#0EA86D;border-radius:50%;animation:p 2.2s infinite}}
@keyframes p{{0%{{box-shadow:0 0 0 0 rgba(14,168,109,.4)}}70%{{box-shadow:0 0 0 9px rgba(14,168,109,0)}}100%{{box-shadow:0 0 0 0 rgba(14,168,109,0)}}}}
.input-wrap{{position:relative;margin-bottom:18px}}
.input-wrap svg{{position:absolute;left:14px;top:50%;transform:translateY(-50%);pointer-events:none}}
input{{
    width:100%;padding:14px 14px 14px 44px;
    border:2px solid #DDE2EF;border-radius:12px;
    font-size:15px;font-family:'DM Sans',sans-serif;color:#0D1117;outline:none;
    transition:border-color .2s,box-shadow .2s;background:#F7F9FC;
}}
input:focus{{border-color:#2F6FED;box-shadow:0 0 0 4px rgba(47,111,237,.12);background:white}}
input::placeholder{{color:#A0AABC}}
.btn{{
    width:100%;padding:15px;
    background:linear-gradient(135deg,#2F6FED,#4F46E5);
    color:white;border:none;border-radius:12px;
    font-family:'Syne',sans-serif;font-weight:700;font-size:16px;
    cursor:pointer;transition:all .22s;letter-spacing:-.01em;
    box-shadow:0 4px 18px rgba(47,111,237,.4);
    position:relative;overflow:hidden;
}}
.btn:hover{{transform:translateY(-2px);box-shadow:0 8px 28px rgba(47,111,237,.5)}}
.btn:active{{transform:translateY(0)}}
.footer{{text-align:center;font-size:11px;color:#A0AABC;margin-top:20px}}
</style>
</head>
<body>
<div class="bg-orb" style="width:400px;height:400px;background:rgba(47,111,237,.15);top:-100px;right:-100px"></div>
<div class="bg-orb" style="width:300px;height:300px;background:rgba(79,70,229,.12);bottom:-80px;left:-80px"></div>
<div class="card">
    <div class="accent-bar"></div>
    <div style="text-align:center">
        <div style="width:60px;height:60px;background:linear-gradient(135deg,#EFF4FF,#C7D9FC);border-radius:16px;margin:0 auto 12px;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 16px rgba(47,111,237,.2)">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2F6FED" stroke-width="2" stroke-linecap="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
        </div>
        <div class="live-badge"><span class="dot"></span>Gate Open</div>
    </div>
    <div class="subject" style="text-align:center">{sess['subject']}</div>
    <div class="sub-label" style="text-align:center">Secure Attendance Check-in · AttendIQ</div>

    <form action="/mark" method="POST" onsubmit="onSub(this)">
        <input type="hidden" name="token" value="{token}">
        <input type="hidden" id="fp" name="fp">
        <div class="input-wrap">
            <svg width="18" height="18" fill="none" stroke="#A0AABC" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M10 6H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V8a2 2 0 00-2-2h-5m-4 0V5a2 2 0 114 0v1m-4 0a2 2 0 104 0m-5 8a2 2 0 100-4 2 2 0 000 4z"/>
            </svg>
            <input type="text" name="student_id" id="sid" placeholder="Enter your Student ID"
                   required autocomplete="off" autocapitalize="off">
        </div>
        <button type="submit" class="btn" id="btn">Confirm Attendance →</button>
    </form>
    <div class="footer">AttendIQ Enterprise · IIT Delhi</div>
</div>
<script>
document.getElementById('fp').value=navigator.userAgent+'|'+screen.width+'x'+screen.height+'|'+navigator.language;
document.getElementById('sid').focus();
function onSub(f){{
    const b=document.getElementById('btn');
    b.innerHTML='<svg width="18" height="18" fill="none" stroke="white" stroke-width="2.5" viewBox="0 0 24 24" style="animation:spin .7s linear infinite"><path stroke-linecap="round" d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"/></svg> Verifying…';
    b.disabled=true; b.style.background='linear-gradient(135deg,#6B7280,#4B5563)';
}}
</script>
<style>@keyframes spin{{to{{transform:rotate(360deg)}}}}</style>
</body>
</html>""")

    return """<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans&display=swap" rel="stylesheet">
<style>body{font-family:'DM Sans',sans-serif;background:#0D1117;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;text-align:center}
h2{font-family:'Syne',sans-serif;font-weight:800;font-size:22px;color:white;margin:12px 0 8px}p{color:#8A94A6;font-size:14px}</style>
</head><body><div><div style="font-size:40px">⚠️</div><h2>Session Expired</h2><p>Please scan a valid QR code.</p></div></body></html>"""


@app.route('/mark', methods=['POST'])
def mark_attendance():
    token = request.form.get('token')
    student_id = request.form.get('student_id', '').strip()
    fp = request.form.get('fp', 'unknown')

    with get_db() as conn:
        sess = conn.execute("SELECT * FROM class_sessions WHERE token=? AND is_active=1", (token,)).fetchone()
        if not sess:
            return """<html><body style="font-family:sans-serif;text-align:center;padding:60px;background:#FEF2F2;color:#E53935">
            <h2>Session Closed</h2><p>This gate is no longer active.</p></body></html>""", 400

        student = conn.execute("SELECT * FROM students WHERE student_id=?", (student_id,)).fetchone()
        if not student:
            return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans&display=swap" rel="stylesheet">
<style>body{{font-family:'DM Sans',sans-serif;background:#0D1117;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;padding:20px}}
.c{{background:white;border-radius:20px;padding:36px;max-width:340px;width:100%;text-align:center;box-shadow:0 24px 60px rgba(0,0,0,.4)}}
h2{{font-family:'Syne',sans-serif;color:#E53935;margin:10px 0 8px}}p{{color:#8A94A6;font-size:13.5px;margin-bottom:20px}}
a{{background:#EFF4FF;color:#2F6FED;text-decoration:none;padding:10px 22px;border-radius:9px;font-weight:600;font-size:13px;display:inline-block}}</style>
</head><body><div class="c"><div style="font-size:36px">🔍</div>
<h2>ID Not Found</h2><p>Student ID <strong>"{student_id}"</strong> is not registered.</p>
<a href="/?token={token}">← Try Again</a></div></body></html>""", 400

        try:
            conn.execute("INSERT INTO attendance (student_id, session_token, timestamp, device_hash) VALUES (?,?,?,?)",
                         (student_id, token, datetime.datetime.now(), fp))
            conn.commit()
        except sqlite3.IntegrityError:
            return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans&display=swap" rel="stylesheet">
<style>body{{font-family:'DM Sans',sans-serif;background:#0D1117;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;padding:20px}}
.c{{background:white;border-radius:20px;padding:36px;max-width:340px;width:100%;text-align:center;box-shadow:0 24px 60px rgba(0,0,0,.4)}}
h2{{font-family:'Syne',sans-serif;color:#F59E0B;margin:10px 0 8px}}p{{color:#8A94A6;font-size:13.5px}}</style>
</head><body><div class="c"><div style="font-size:36px">ℹ️</div>
<h2>Already Marked</h2><p>Attendance for <strong>{student['name']}</strong> is already recorded for this session.</p>
</div></body></html>"""

    detect_anomalies()
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'DM Sans',sans-serif;background:linear-gradient(135deg,#0D1117,#1A2540);
     display:flex;align-items:center;justify-content:center;min-height:100vh;padding:20px;overflow:hidden}}
.bg{{position:fixed;border-radius:50%;filter:blur(60px);pointer-events:none}}
.card{{background:rgba(255,255,255,.97);border-radius:24px;padding:44px 36px;
       box-shadow:0 24px 80px rgba(0,0,0,.4);max-width:380px;width:100%;text-align:center;
       animation:cardIn .65s cubic-bezier(.16,1,.3,1) both}}
@keyframes cardIn{{from{{opacity:0;transform:translateY(24px) scale(.95)}}to{{opacity:1;transform:translateY(0) scale(1)}}}}
h2{{font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:#0EA86D;margin:16px 0 8px}}
p{{font-size:13.5px;color:#8A94A6;line-height:1.65}}
.name{{font-weight:700;color:#0D1117;font-size:15px;margin-bottom:4px}}
canvas#confetti{{position:fixed;inset:0;pointer-events:none;z-index:100}}
@keyframes checkDraw{{to{{stroke-dashoffset:0}}}}
@keyframes ringScale{{from{{transform:scale(0) rotate(-90deg)}}to{{transform:scale(1) rotate(-90deg)}}}}
.ring-svg{{animation:ringScale .5s cubic-bezier(.16,1,.3,1) both}}
.check-path{{stroke-dasharray:80;stroke-dashoffset:80;animation:checkDraw .6s .45s cubic-bezier(.16,1,.3,1) forwards}}
.text-wrap{{animation:fadeUp .5s .75s both;opacity:0}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:translateY(0)}}}}
</style>
</head>
<body>
<div class="bg" style="width:350px;height:350px;background:rgba(14,168,109,.12);top:-80px;right:-80px"></div>
<div class="bg" style="width:280px;height:280px;background:rgba(47,111,237,.1);bottom:-60px;left:-60px"></div>
<canvas id="confetti"></canvas>
<div class="card">
    <div style="width:88px;height:88px;margin:0 auto">
        <svg class="ring-svg" viewBox="0 0 88 88" width="88" height="88" style="transform:rotate(-90deg)">
            <circle cx="44" cy="44" r="40" fill="#EAFAF3" stroke="#A3E6CC" stroke-width="2"/>
            <polyline class="check-path" points="26,44 40,58 62,32" fill="none" stroke="#0EA86D" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    </div>
    <div class="text-wrap">
        <h2>Attendance Confirmed!</h2>
        <p class="name">{student['name']}</p>
        <p>Marked present for<br><strong style="color:#0D1117">{sess['subject']}</strong></p>
        <p style="margin-top:18px;font-size:11.5px;color:#C0CADB">You may close this window</p>
    </div>
</div>
<script>
// Confetti burst
(function(){{
    const c=document.getElementById('confetti');
    const ctx=c.getContext('2d');
    c.width=window.innerWidth; c.height=window.innerHeight;
    const particles=[];
    const colors=['#2F6FED','#0EA86D','#F59E0B','#E53935','#7C3AED','#EC4899'];
    for(let i=0;i<120;i++) particles.push({{
        x:c.width/2, y:c.height*.5,
        vx:(Math.random()-0.5)*18,
        vy:-(Math.random()*14+4),
        r:Math.random()*6+3,
        color:colors[Math.floor(Math.random()*colors.length)],
        rot:Math.random()*360, rotV:(Math.random()-.5)*8,
        alpha:1
    }});
    let frame=0;
    function draw(){{
        ctx.clearRect(0,0,c.width,c.height);
        particles.forEach(p=>{{
            p.x+=p.vx; p.y+=p.vy; p.vy+=0.35;
            p.rot+=p.rotV; p.alpha-=0.014;
            if(p.alpha<=0) return;
            ctx.save(); ctx.translate(p.x,p.y); ctx.rotate(p.rot*Math.PI/180);
            ctx.globalAlpha=p.alpha;
            ctx.fillStyle=p.color;
            ctx.fillRect(-p.r/2,-p.r/2,p.r,p.r*1.6);
            ctx.restore();
        }});
        if(frame++<180) requestAnimationFrame(draw);
    }}
    setTimeout(draw, 700);
}})();
</script>
</body></html>"""


# ── SESSION MANAGEMENT ────────────────────────────────────────────────────────
@app.route('/start_session', methods=['POST'])
def start_session():
    subject = request.form.get('subject')
    notes   = request.form.get('notes', '')
    token   = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute("UPDATE class_sessions SET is_active=0")
        conn.execute("INSERT INTO class_sessions (token,created_at,is_active,subject,notes) VALUES (?,?,1,?,?)",
                     (token, datetime.datetime.now(), subject, notes))
        conn.commit()
    flash(f"Session launched: {subject}")
    return redirect(url_for('admin_panel'))

@app.route('/end_session/<token>')
def end_session(token):
    with get_db() as conn:
        conn.execute("UPDATE class_sessions SET is_active=0 WHERE token=?", (token,))
        conn.commit()
    flash("Session closed successfully.")
    return redirect(url_for('admin_panel'))

@app.route('/qr/<token>')
def generate_qr(token):
    url = f"{BASE_URL}/?token={token}"
    qr  = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=10, border=4)
    qr.add_data(url); qr.make(fit=True)
    img = qr.make_image(fill_color="#0D1117", back_color="white")
    buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    return send_file(buf, mimetype='image/png')


# ── STUDENT REGISTRATION ──────────────────────────────────────────────────────
@app.route('/register_student', methods=['GET', 'POST'])
def register_student():
    if request.method == 'POST':
        with get_db() as conn:
            try:
                conn.execute("INSERT INTO students (student_id,name,email,phone,department,year) VALUES (?,?,?,?,?,?)",
                             (request.form.get('student_id'), request.form.get('name'),
                              request.form.get('email'), request.form.get('phone',''),
                              request.form.get('department',''), request.form.get('year',1)))
                conn.commit()
                flash(f"Student {request.form.get('name')} enrolled.")
                return redirect(url_for('manage_students'))
            except sqlite3.IntegrityError:
                flash("Error: Student ID already exists.")

    content = """
<div style="max-width:580px;margin:0 auto">
    <div class="anim" style="margin-bottom:26px">
        <h1 style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;letter-spacing:-.02em;margin-bottom:4px">Enroll Student</h1>
        <p style="color:var(--muted);font-size:13px">Add a new student to the system roster</p>
    </div>
    <div class="card anim d1" style="padding:30px">
        <form method="POST">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
                <div class="form-group">
                    <label class="form-label">Student ID *</label>
                    <input type="text" name="student_id" required class="form-input" placeholder="e.g. STU2024019">
                </div>
                <div class="form-group">
                    <label class="form-label">Full Name *</label>
                    <input type="text" name="name" required class="form-input" placeholder="Full name">
                </div>
                <div class="form-group">
                    <label class="form-label">Email Address</label>
                    <input type="email" name="email" class="form-input" placeholder="student@iitd.ac.in">
                </div>
                <div class="form-group">
                    <label class="form-label">Phone</label>
                    <input type="tel" name="phone" class="form-input" placeholder="+91 98765 43210">
                </div>
                <div class="form-group">
                    <label class="form-label">Department</label>
                    <input type="text" name="department" class="form-input" placeholder="e.g. Computer Science">
                </div>
                <div class="form-group">
                    <label class="form-label">Year</label>
                    <select name="year" class="form-select">
                        <option value="1">1st Year</option>
                        <option value="2">2nd Year</option>
                        <option value="3">3rd Year</option>
                        <option value="4">4th Year</option>
                    </select>
                </div>
            </div>
            <button type="submit" class="btn btn-primary" style="width:100%;justify-content:center;margin-top:8px;padding:12px">
                <svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z"/></svg>
                Save Record
            </button>
        </form>
    </div>
</div>"""
    return page(content, "Enroll Student", "students")


# ── STUDENT LIST ──────────────────────────────────────────────────────────────
@app.route('/manage_students')
def manage_students():
    with get_db() as conn:
        students = conn.execute("SELECT * FROM students ORDER BY name").fetchall()

    rows = ""
    for s in students:
        st = calculate_stats(s['student_id'])
        pct = st['percentage'] if st else None
        risk = st['risk_level'] if st else None
        pct_html = f"""<div style="display:flex;align-items:center;gap:8px">
            <div style="width:64px"><div class="prog-bar"><div class="prog-fill" data-prog="{pct}"
                style="background:{'linear-gradient(90deg,var(--green),#34EEB0)' if pct and pct>=75 else 'linear-gradient(90deg,var(--red),#FF6B6B)'};width:0%"></div></div></div>
            <span class="mono" style="font-size:12px;font-weight:600;color:{'var(--green)' if pct and pct>=75 else 'var(--red)' if pct else 'var(--muted)'}">{'%.1f%%' % pct if pct is not None else '—'}</span>
        </div>""" if pct is not None else '<span style="color:var(--muted);font-size:13px">—</span>'

        dept_colors = {'Computer Science':'badge-blue','Data Science':'badge-violet','AI & ML':'badge-green'}
        dept_badge = dept_colors.get(s['department'],'badge-gray')

        rows += f"""<tr class="row-link" data-searchrow onclick="window.location='/student/{s['student_id']}'">
            <td>
                <div style="display:flex;align-items:center;gap:10px">
                    <div class="avatar" style="background:linear-gradient(135deg,#EFF4FF,#C7D9FC);color:var(--blue)">
                        {s['name'][0].upper()}
                    </div>
                    <div>
                        <div style="font-weight:600;font-size:13px">{s['name']}</div>
                        <div class="mono" style="font-size:10px;color:var(--muted)">{s['student_id']}</div>
                    </div>
                </div>
            </td>
            <td style="font-size:13px;color:var(--text2)">{s['email'] or '—'}</td>
            <td><span class="badge {dept_badge}" style="font-size:10.5px">{s['department'] or '—'}</span></td>
            <td>{pct_html}</td>
            <td>{'<span class="badge badge-red">⚠ Critical</span>' if risk=='CRITICAL' else '<span class="badge badge-green">✓ Safe</span>' if risk=='SAFE' else '<span class="badge badge-gray">No data</span>'}</td>
            <td style="text-align:right"><span class="badge badge-blue" style="cursor:pointer">View →</span></td>
        </tr>"""

    content = f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:26px" class="anim">
    <div>
        <h1 style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;letter-spacing:-.02em;margin-bottom:4px">Students</h1>
        <p style="color:var(--muted);font-size:13px">{len(students)} students enrolled</p>
    </div>
    <div style="display:flex;gap:10px">
        <div class="search-wrap">
            <svg width="14" height="14" fill="none" stroke="#94A3B8" stroke-width="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path stroke-linecap="round" d="m21 21-4.35-4.35"/></svg>
            <input id="tableSearch" class="form-input" placeholder="Search name, ID…" style="width:210px;padding:7px 10px 7px 30px">
        </div>
        <a href="/register_student" class="btn btn-primary">+ Enroll</a>
        <a href="/bulk_import" class="btn btn-secondary">Import CSV</a>
    </div>
</div>
<div class="card anim d1" style="overflow:hidden">
    <div style="overflow-x:auto">
        <table class="data-table">
            <thead><tr><th>Student</th><th>Email</th><th>Department</th><th>Attendance</th><th>Status</th><th style="text-align:right">Profile</th></tr></thead>
            <tbody>{rows or '<tr><td colspan="6" style="text-align:center;padding:60px;color:var(--muted)">No students yet. <a href="/register_student" style="color:var(--blue)">Enroll one →</a></td></tr>'}</tbody>
        </table>
    </div>
</div>"""
    return page(content, "Students", "students")


# ── SESSIONS LIST ─────────────────────────────────────────────────────────────
@app.route('/manage_sessions')
def manage_sessions():
    with get_db() as conn:
        sessions = conn.execute("SELECT cs.*, COUNT(a.id) as checkins FROM class_sessions cs LEFT JOIN attendance a ON cs.token=a.session_token GROUP BY cs.token ORDER BY cs.created_at DESC").fetchall()

    rows = ''.join([f"""
    <tr>
        <td class="mono" style="font-size:11.5px;color:var(--muted)">{s['created_at'][:16]}</td>
        <td style="font-size:13px;font-weight:600;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{s['subject']}</td>
        <td>{'<span class="badge badge-green"><span class="live-dot" style="width:6px;height:6px;margin-right:3px;display:inline-block"></span>Active</span>' if s['is_active'] else '<span class="badge badge-gray">Closed</span>'}</td>
        <td><span class="badge badge-blue">{s['checkins']} checked in</span></td>
        <td class="mono" style="font-size:10.5px;color:var(--muted)">{s['token'][:16]}…</td>
        <td style="text-align:right">
            {'<a href="/end_session/' + s["token"] + '" class="btn btn-danger btn-sm">End</a>' if s['is_active'] else ''}
        </td>
    </tr>""" for s in sessions])

    content = f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:26px" class="anim">
    <div>
        <h1 style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;letter-spacing:-.02em;margin-bottom:4px">Sessions</h1>
        <p style="color:var(--muted);font-size:13px">{len(sessions)} sessions recorded</p>
    </div>
</div>
<div class="card anim d1" style="overflow:hidden">
    <div style="overflow-x:auto">
        <table class="data-table">
            <thead><tr><th>Date</th><th>Subject</th><th>Status</th><th>Attendance</th><th>Token</th><th style="text-align:right">Actions</th></tr></thead>
            <tbody>{rows or '<tr><td colspan="6" style="text-align:center;padding:60px;color:var(--muted)">No sessions yet</td></tr>'}</tbody>
        </table>
    </div>
</div>"""
    return page(content, "Sessions", "sessions")


# ── REPORTS ───────────────────────────────────────────────────────────────────
@app.route('/reports')
def reports():
    with get_db() as conn:
        students = conn.execute("SELECT * FROM students ORDER BY name").fetchall()
        by_subject = conn.execute("""SELECT cs.subject,
            COUNT(DISTINCT cs.token) sessions,
            COUNT(a.id) total_checkins,
            COUNT(DISTINCT a.student_id) unique_students
            FROM class_sessions cs LEFT JOIN attendance a ON cs.token=a.session_token
            GROUP BY cs.subject ORDER BY total_checkins DESC""").fetchall()

    report_rows = ""
    for s in students:
        st = calculate_stats(s['student_id'])
        if not st: continue
        report_rows += f"""
        <tr data-searchrow>
            <td style="font-weight:600;font-size:13px">{s['name']}</td>
            <td class="mono" style="font-size:11px;color:var(--muted)">{s['student_id']}</td>
            <td style="font-size:13px;color:var(--text2)">{s['department'] or '—'}</td>
            <td><div style="display:flex;align-items:center;gap:8px">
                <div style="width:72px"><div class="prog-bar"><div class="prog-fill" data-prog="{st['percentage']}"
                    style="background:{'linear-gradient(90deg,var(--green),#34EEB0)' if st['percentage']>=75 else 'linear-gradient(90deg,var(--red),#FF6B6B)'};width:0%"></div></div></div>
                <span class="mono" style="font-size:12px;font-weight:700;color:{'var(--green)' if st['percentage']>=75 else 'var(--red)'}">{st['percentage']}%</span>
            </div></td>
            <td class="mono" style="font-size:12px">{st['present']} / {st['total']}</td>
            <td>{'<span class="badge badge-red">Critical</span>' if st['risk_level']=='CRITICAL' else '<span class="badge badge-green">Safe</span>'}</td>
            <td class="mono" style="font-size:12px;color:{'var(--red)' if st['sessions_needed']>0 else 'var(--green)'}">
                {'⚠ ' + str(st['sessions_needed']) + ' needed' if st['sessions_needed']>0 else '✓ OK'}
            </td>
        </tr>"""

    subject_rows = ''.join([f"""
    <tr>
        <td style="font-weight:600;font-size:13px">{r['subject']}</td>
        <td class="mono" style="font-size:12px">{r['sessions']}</td>
        <td class="mono" style="font-size:12px">{r['total_checkins']}</td>
        <td class="mono" style="font-size:12px">{r['unique_students']}</td>
        <td class="mono" style="font-size:12px;color:var(--blue)">{round(r['total_checkins']/r['sessions'],1) if r['sessions'] else 0} avg</td>
    </tr>""" for r in by_subject])

    at_risk_count  = sum(1 for s in students if calculate_stats(s['student_id']) and calculate_stats(s['student_id'])['risk_level']=='CRITICAL')
    avg_att = round(sum(calculate_stats(s['student_id'])['percentage'] for s in students if calculate_stats(s['student_id'])) / max(1, sum(1 for s in students if calculate_stats(s['student_id']))), 1)

    content = f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:26px" class="anim">
    <div>
        <h1 style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;letter-spacing:-.02em;margin-bottom:4px">Reports</h1>
        <p style="color:var(--muted);font-size:13px">Full attendance summary &amp; subject breakdown</p>
    </div>
    <a href="/export_excel" class="btn btn-primary">↓ Export CSV</a>
</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px">
    <div class="card anim d1">
        <div class="card-header"><div class="section-title">Subject Report</div></div>
        <div style="overflow-x:auto">
            <table class="data-table">
                <thead><tr><th>Subject</th><th>Sessions</th><th>Total Check-ins</th><th>Unique Students</th><th>Avg/Session</th></tr></thead>
                <tbody>{subject_rows or '<tr><td colspan="5" style="text-align:center;padding:30px;color:var(--muted)">No data yet</td></tr>'}</tbody>
            </table>
        </div>
    </div>
    <div class="card anim d2">
        <div class="card-header"><div class="section-title">Quick Stats</div></div>
        <div style="padding:20px;display:grid;grid-template-columns:1fr 1fr;gap:14px">
            {''.join([f"""<div style="background:var(--bg);border-radius:12px;padding:18px;border:1px solid var(--border2)">
                <div class="mono" style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">{label}</div>
                <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:{col}">{val}</div>
            </div>""" for label, val, col in [
                ('Total Students', len(students), 'var(--text)'),
                ('At Risk', at_risk_count, 'var(--red)'),
                ('Avg Attendance', f'{avg_att}%', 'var(--blue)'),
                ('Total Sessions', sum(r['sessions'] for r in by_subject) if by_subject else 0, 'var(--text)'),
            ]])}
        </div>
    </div>
</div>

<div class="card anim d3">
    <div class="card-header">
        <div><div class="section-title">Full Student Report</div></div>
        <div class="search-wrap">
            <svg width="14" height="14" fill="none" stroke="#94A3B8" stroke-width="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path stroke-linecap="round" d="m21 21-4.35-4.35"/></svg>
            <input id="tableSearch" class="form-input" placeholder="Search…" style="width:180px;padding:7px 10px 7px 30px">
        </div>
    </div>
    <div style="overflow-x:auto">
        <table class="data-table">
            <thead><tr><th>Name</th><th>ID</th><th>Dept</th><th>Attendance</th><th>Present/Total</th><th>Status</th><th>Action Needed</th></tr></thead>
            <tbody>{report_rows or '<tr><td colspan="7" style="text-align:center;padding:40px;color:var(--muted)">No data yet</td></tr>'}</tbody>
        </table>
    </div>
</div>"""
    return page(content, "Reports", "reports")


# ── BULK IMPORT ───────────────────────────────────────────────────────────────
@app.route('/bulk_import', methods=['GET', 'POST'])
def bulk_import():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                with get_db() as conn:
                    added = 0
                    for _, row in df.iterrows():
                        try:
                            conn.execute("INSERT INTO students (student_id,name,email,department) VALUES (?,?,?,?)",
                                         (str(row['student_id']), row['name'],
                                          row.get('email',''), row.get('department','')))
                            added += 1
                        except sqlite3.IntegrityError:
                            continue
                    conn.commit()
                flash(f"Imported {added} students successfully.")
            except Exception as e:
                flash(f"Error: {e}")
        return redirect(url_for('manage_students'))

    content = """
<div style="max-width:580px;margin:0 auto">
    <div class="anim" style="margin-bottom:26px">
        <h1 style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;letter-spacing:-.02em;margin-bottom:4px">Bulk Import</h1>
        <p style="color:var(--muted);font-size:13px">Upload a CSV to enroll multiple students at once</p>
    </div>
    <div class="card anim d1" style="padding:30px">
        <div style="background:var(--blue-lt);border:1px solid var(--blue-mid);border-radius:10px;padding:14px 16px;margin-bottom:22px;font-size:13px;color:#1D4ED8">
            <strong>Required columns:</strong>
            <code style="background:rgba(47,111,237,.1);padding:2px 7px;border-radius:5px;font-family:'JetBrains Mono',monospace;font-size:12px">student_id, name, email</code>
            &nbsp; Optional: <code style="background:rgba(47,111,237,.1);padding:2px 7px;border-radius:5px;font-family:'JetBrains Mono',monospace;font-size:12px">department</code>
        </div>
        <form method="POST" enctype="multipart/form-data">
            <div id="dropzone" style="border:2px dashed var(--border);border-radius:14px;padding:52px 20px;text-align:center;
                 cursor:pointer;transition:all .22s;background:var(--bg);position:relative;margin-bottom:18px"
                 ondragover="event.preventDefault();this.style.borderColor='var(--blue)';this.style.background='var(--blue-lt)'"
                 ondragleave="this.style.borderColor='var(--border)';this.style.background='var(--bg)'"
                 ondrop="handleDrop(event)">
                <input type="file" name="file" accept=".csv" required id="fileInput"
                       style="position:absolute;inset:0;opacity:0;cursor:pointer" onchange="updateZone(this)">
                <div id="uploadIcon" style="width:52px;height:52px;background:var(--border);border-radius:14px;display:flex;align-items:center;justify-content:center;margin:0 auto 14px">
                    <svg width="24" height="24" fill="none" stroke="var(--muted)" stroke-width="1.8" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
                    </svg>
                </div>
                <p id="dropText" style="font-weight:600;color:var(--text2);font-size:14px">Drop CSV here or click to browse</p>
                <p style="font-size:12px;color:var(--muted);margin-top:5px">.csv files only · Max 10MB</p>
            </div>
            <button type="submit" class="btn btn-primary" style="width:100%;justify-content:center;padding:12px">
                <svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                Import Students
            </button>
        </form>
    </div>
</div>
<script>
function updateZone(inp){
    if(inp.files[0]){
        document.getElementById('dropText').textContent='✓ '+inp.files[0].name;
        document.getElementById('dropText').style.color='var(--green)';
        document.getElementById('uploadIcon').style.background='var(--green-lt)';
        document.getElementById('dropzone').style.borderColor='#A3E6CC';
        document.getElementById('dropzone').style.background='var(--green-lt)';
    }
}
function handleDrop(e){
    e.preventDefault();
    const fi=document.getElementById('fileInput');
    if(e.dataTransfer.files[0]){fi.files=e.dataTransfer.files;updateZone(fi);}
    document.getElementById('dropzone').style.borderColor='var(--border)';
}
</script>"""
    return page(content, "Bulk Import", "import")


# ── SETTINGS ──────────────────────────────────────────────────────────────────
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        with get_db() as conn:
            for key in ['threshold', 'total_sessions', 'org_name']:
                val = request.form.get(key)
                if val:
                    conn.execute("INSERT OR REPLACE INTO app_settings (key,value) VALUES (?,?)", (key, val))
            conn.commit()
        flash("Settings saved.")
        return redirect(url_for('settings'))

    thresh = get_setting('threshold', '75')
    total  = get_setting('total_sessions', '40')
    org    = get_setting('org_name', 'My Institution')

    content = f"""
<div style="max-width:580px;margin:0 auto">
    <div class="anim" style="margin-bottom:26px">
        <h1 style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;letter-spacing:-.02em;margin-bottom:4px">Settings</h1>
        <p style="color:var(--muted);font-size:13px">Configure system-wide parameters</p>
    </div>
    <div class="card anim d1" style="padding:30px;margin-bottom:16px">
        <form method="POST">
            <div class="form-group">
                <label class="form-label">Organisation Name</label>
                <input type="text" name="org_name" value="{org}" class="form-input">
                <div class="form-hint">Shown in topbar</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
                <div class="form-group">
                    <label class="form-label">Attendance Threshold (%)</label>
                    <input type="number" name="threshold" value="{thresh}" min="0" max="100" class="form-input">
                    <div class="form-hint">Students below this are flagged</div>
                </div>
                <div class="form-group">
                    <label class="form-label">Total Sessions (Estimate)</label>
                    <input type="number" name="total_sessions" value="{total}" min="1" class="form-input">
                    <div class="form-hint">Used for projection</div>
                </div>
            </div>
            <button type="submit" class="btn btn-primary" style="width:100%;justify-content:center;padding:12px">
                Save Settings
            </button>
        </form>
    </div>

    <div class="card anim d2" style="padding:26px;background:var(--bg)">
        <div class="section-title" style="margin-bottom:14px">Network Info</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
            <div style="background:white;border:1px solid var(--border);border-radius:10px;padding:14px">
                <div class="mono" style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px">Local IP</div>
                <div class="mono" style="font-size:14px;font-weight:600">{LOCAL_IP}</div>
            </div>
            <div style="background:white;border:1px solid var(--border);border-radius:10px;padding:14px">
                <div class="mono" style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px">Check-in URL</div>
                <div class="mono" style="font-size:12px;color:var(--blue);word-break:break-all">{BASE_URL}</div>
            </div>
        </div>
        <div style="margin-top:12px;padding:12px 14px;background:#FFFBEB;border:1px solid #FDE68A;border-radius:9px;font-size:12.5px;color:#78350F">
            <strong>Phone not loading?</strong> Make sure your phone and computer are on the same Wi-Fi network.
        </div>
    </div>
</div>"""
    return page(content, "Settings", "settings")


# ── EXPORT ────────────────────────────────────────────────────────────────────
@app.route('/export_excel')
def export_excel():
    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql_query("""
            SELECT s.student_id, s.name, s.email, s.department, cs.subject,
                   a.timestamp, CASE WHEN a.is_anomaly=1 THEN 'Flagged' ELSE 'Valid' END as status
            FROM attendance a JOIN students s ON a.student_id=s.student_id
            JOIN class_sessions cs ON a.session_token=cs.token ORDER BY a.timestamp DESC""", conn)
    out = io.StringIO()
    df.to_csv(out, index=False)
    out.seek(0)
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=attendiq_export.csv"})


if __name__ == '__main__':
    init_db()
    seed_dummy_data()
    print(f"\n{'━'*56}")
    print(f"  AttendIQ Enterprise v2  ·  Enhanced Edition")
    print(f"  Admin      → {BASE_URL}/admin")
    print(f"  Analytics  → {BASE_URL}/analytics")
    print(f"  Reports    → {BASE_URL}/reports")
    print(f"\n  Seeded with 18 Indian student profiles & historical data.")
    print(f"{'━'*56}\n")
    app.run(host='0.0.0.0', port=PORT, debug=True)