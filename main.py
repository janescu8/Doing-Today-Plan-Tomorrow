# main_sql_drive.py
import os, io, uuid, time, sqlite3, datetime
import streamlit as st
import pandas as pd

# --- Optional summarizer (no external API) ---
try:
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
except Exception:
    LsaSummarizer = None

# --- Google Drive client ---
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# ----------------------------
# Secrets expected:
# st.secrets["google_auth"] : service account JSON
# st.secrets["google_drive"]["db_folder_id"]
# st.secrets["google_drive"]["attachments_folder_id"]
# st.secrets["google_drive"]["db_filename"] (e.g., "journal.sqlite")
# ----------------------------

DB_PATH = "journal.sqlite"

def drive_service():
    creds = Credentials.from_service_account_info(
        st.secrets["google_auth"],
        scopes=[
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/drive.file",
        ],
    )
    return build("drive", "v3", credentials=creds)

def find_file_in_folder(svc, name, folder_id):
    q = f"name = '{name}' and '{folder_id}' in parents and trashed = false"
    resp = svc.files().list(q=q, fields="files(id, name)").execute()
    files = resp.get("files", [])
    return files[0] if files else None

def download_db(local_path):
    svc = drive_service()
    folder_id = st.secrets["google_drive"]["db_folder_id"]
    fname = st.secrets["google_drive"]["db_filename"]
    f = find_file_in_folder(svc, fname, folder_id)
    if not f:
        return False
    req = svc.files().get_media(fileId=f["id"])
    with open(local_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    return True

def upload_db(local_path):
    svc = drive_service()
    folder_id = st.secrets["google_drive"]["db_folder_id"]
    fname = st.secrets["google_drive"]["db_filename"]
    f = find_file_in_folder(svc, fname, folder_id)

    media = MediaIoBaseUpload(io.FileIO(local_path, "rb"), mimetype="application/octet-stream", resumable=True)
    file_meta = {"name": fname, "parents": [folder_id]}
    if f:
        svc.files().update(fileId=f["id"], media_body=media).execute()
    else:
        svc.files().create(body=file_meta, media_body=media, fields="id").execute()

def upload_attachment(file, user) -> str:
    """Upload a file to Drive attachments folder, return fileId."""
    svc = drive_service()
    folder_id = st.secrets["google_drive"]["attachments_folder_id"]
    unique_name = f"{user}-{int(time.time())}-{file.name}"
    file_meta = {"name": unique_name, "parents": [folder_id]}
    media = MediaIoBaseUpload(io.BytesIO(file.getvalue()), mimetype=file.type or "application/octet-stream", resumable=True)
    created = svc.files().create(body=file_meta, media_body=media, fields="id").execute()
    return created["id"]

def public_view_url(file_id: str) -> str:
    # For private apps you can manage permissions; this direct URL usually renders images.
    return f"https://drive.google.com/uc?id={file_id}"

# ----------------------------
# DB bootstrap & helpers
# ----------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entries (
  id TEXT PRIMARY KEY,
  user TEXT NOT NULL,
  date TEXT NOT NULL,
  what TEXT,
  meaningful TEXT,
  mood INTEGER,
  choice TEXT,
  no_repeat TEXT,
  plans TEXT,
  tags TEXT,
  summary TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS attachments (
  id TEXT PRIMARY KEY,
  entry_id TEXT NOT NULL,
  drive_file_id TEXT NOT NULL,
  mime TEXT,
  FOREIGN KEY(entry_id) REFERENCES entries(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  entry_id TEXT NOT NULL,
  text TEXT NOT NULL,
  is_done INTEGER DEFAULT 0,
  FOREIGN KEY(entry_id) REFERENCES entries(id) ON DELETE CASCADE
);
"""

def ensure_db():
    if not os.path.exists(DB_PATH):
        downloaded = download_db(DB_PATH)
        if not downloaded:
            # first run: create empty db locally & upload it
            conn = sqlite3.connect(DB_PATH)
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            conn.close()
            upload_db(DB_PATH)

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def summarize_text(text, max_sentences=2):
    if not text or not text.strip():
        return ""
    if LsaSummarizer is None:
        # fallback: first 300 chars of first line
        return text.split("\n")[0][:300]
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summ = LsaSummarizer()
        sents = summ(parser.document, max_sentences)
        return " ".join(str(s) for s in sents)
    except Exception:
        return text.split("\n")[0][:300]

def save_entry_to_db(user, date, what, meaningful, mood, choice, no_repeat, plans, tags, uploaded_files):
    entry_id = str(uuid.uuid4())
    tags_str = ", ".join([t.strip() for t in (tags or "").split(",") if t.strip()])
    summary = summarize_text(what)

    conn = get_conn()
    c = conn.cursor()
    c.execute("""INSERT INTO entries (id,user,date,what,meaningful,mood,choice,no_repeat,plans,tags,summary)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
              (entry_id, user, date, what, meaningful, int(mood) if mood else None, choice, no_repeat, plans, tags_str, summary))

    # attachments
    for f in uploaded_files or []:
        file_id = upload_attachment(f, user)
        c.execute("""INSERT INTO attachments (id, entry_id, drive_file_id, mime)
                     VALUES (?,?,?,?)""", (str(uuid.uuid4()), entry_id, file_id, f.type or ""))

    # tasks (one per line)
    for line in (plans or "").split("\n"):
        t = line.strip()
        if t:
            c.execute("""INSERT INTO tasks (id, entry_id, text, is_done)
                         VALUES (?,?,?,0)""", (str(uuid.uuid4()), entry_id, t))

    conn.commit()
    conn.close()
    # sync DB up to Drive
    upload_db(DB_PATH)
    return entry_id

def load_entry_bundle(user, limit=10):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""SELECT id, user, date, what, meaningful, mood, choice, no_repeat, plans, tags, summary, created_at
                 FROM entries WHERE user=? ORDER BY date DESC, created_at DESC LIMIT ?""", (user, limit))
    cols = [d[0] for d in c.description]
    entries = [dict(zip(cols, row)) for row in c.fetchall()]

    for e in entries:
        eid = e["id"]
        c.execute("SELECT id, drive_file_id, mime FROM attachments WHERE entry_id=?", (eid,))
        e["attachments"] = [{"id": aid, "file_id": fid, "url": public_view_url(fid), "mime": mime} for (aid, fid, mime) in c.fetchall()]
        c.execute("SELECT id, text, is_done FROM tasks WHERE entry_id=?", (eid,))
        e["tasks"] = [{"id": tid, "text": t, "is_done": bool(done)} for (tid, t, done) in c.fetchall()]
    conn.close()
    return entries

def update_task_done(task_id: str, is_done: bool):
    conn = get_conn()
    conn.execute("UPDATE tasks SET is_done=? WHERE id=?", (1 if is_done else 0, task_id))
    conn.commit()
    conn.close()
    upload_db(DB_PATH)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Sanny’s Diary", page_icon="🌀", layout="centered")
st.title("🌀 迷惘但想搞懂的我 / Lost but Learning")
st.caption("SQLite stored in Google Drive • Attachments in Drive • Summaries via sumy")

ensure_db()

# Simple user selection (single-user friendly)
user = st.sidebar.text_input("使用者 / User", value="Sanny")
st.sidebar.info("資料永久保存：DB 與附件都存放在 Google Drive。")

today = datetime.date.today().strftime("%Y-%m-%d")
with st.form("new_entry"):
    st.subheader("新增日記 / New Entry")
    what = st.text_area("📌 今天你做了什麼 / What did you do today?", height=140)
    meaningful = st.text_area("🎯 今天有感覺的事 / What felt meaningful today?")
    mood = st.slider("📊 今天整體感受 (1-10)", 1, 10, 5)
    choice = st.text_area("🧠 是自主選擇嗎？/ Was it your choice?")
    no_repeat = st.text_area("🚫 今天最不想再來的事 / What you wouldn't repeat?")
    plans = st.text_area("🌱 明天想做什麼（每行一個任務） / Plans for tomorrow (one per line)")
    tags = st.text_input("🏷️ 標籤 / Tags (comma-separated)")
    uploaded_files = st.file_uploader("📎 附加檔案 / Attachments (image or audio)", accept_multiple_files=True)
    submitted = st.form_submit_button("提交 / Submit")
    if submitted:
        save_entry_to_db(user, today, what, meaningful, mood, choice, no_repeat, plans, tags, uploaded_files)
        st.success("已送出！資料與檔案已同步到 Google Drive ✅")

st.divider()
st.subheader("📜 歷史紀錄（最近10筆） / Recent Entries")

entries = load_entry_bundle(user, limit=10)
if not entries:
    st.info("尚無紀錄。")
else:
    for e in entries:
        st.markdown(f"**🗓️ {e['date']}** — **Mood:** {e['mood'] if e['mood'] is not None else '-'} /10")
        st.markdown(f"**What:** {e['what'] or ''}")
        if e["meaningful"]:
            st.markdown(f"**Meaningful:** {e['meaningful']}")
        if e["summary"]:
            with st.expander("🧾 Summary (auto)"):
                st.write(e["summary"])

        if e["attachments"]:
            st.write("**Attachments:**")
            for a in e["attachments"]:
                if (a["mime"] or "").startswith("image/"):
                    st.image(a["url"], width=240)
                elif (a["mime"] or "").startswith("audio/"):
                    st.audio(a["url"])

        # Tasks
        if e["tasks"]:
            st.write("**Tomorrow’s Tasks:**")
            for t in e["tasks"]:
                new_val = st.checkbox(t["text"], value=t["is_done"], key=f"task-{t['id']}")
                if new_val != t["is_done"]:
                    update_task_done(t["id"], new_val)

st.divider()
st.subheader("📤 匯出 / Export")

if entries:
    # Basic CSV export (entries-only)
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM entries WHERE user = ? ORDER BY date DESC, created_at DESC", conn, params=(user,))
    conn.close()
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下載 CSV (entries)", csv, file_name="entries.csv", mime="text/csv")

st.caption("Note: For private images, manage Drive permissions. For multi-user editing, consider adding a simple lock or moving to Postgres.")
