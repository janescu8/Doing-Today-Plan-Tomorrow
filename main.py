# main.py — Diary app with section navigator, proxy media, images/audio/videos split, history=5
import os, io, uuid, time, sqlite3, datetime
import streamlit as st
import pandas as pd

# --- Optional summarizer (no external LLM) ---
try:
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
except Exception:
    LsaSummarizer = None

# --- Optional OpenAI LLM for monthly summary (fallback to sumy) ---
OPENAI_AVAILABLE = False
try:
    import openai
    api_key = None
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        api_key = st.secrets["openai"]["api_key"]
    elif os.getenv("OPENAI_API_KEY"):
        api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai.api_key = api_key
        OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# --- Google Drive client (non-resumable uploads for reliability) ---
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

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
    resp = svc.files().list(
        q=q, fields="files(id, name)",
        includeItemsFromAllDrives=True, supportsAllDrives=True,
        corpora="allDrives", spaces="drive",
    ).execute()
    files = resp.get("files", [])
    return files[0] if files else None

def get_drive_db_file():
    svc = drive_service()
    folder_id = st.secrets["google_drive"]["db_folder_id"]
    fname = st.secrets["google_drive"]["db_filename"]
    return find_file_in_folder(svc, fname, folder_id)

def download_db(local_path):
    svc = drive_service()
    f = get_drive_db_file()
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
    media = MediaFileUpload(local_path, mimetype="application/octet-stream", resumable=False)
    meta = {"name": fname, "parents": [folder_id]}
    if f:
        svc.files().update(fileId=f["id"], media_body=media, supportsAllDrives=True).execute()
    else:
        svc.files().create(body=meta, media_body=media, fields="id", supportsAllDrives=True).execute()

def make_public_read(file_id: str):
    svc = drive_service()
    try:
        svc.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
            supportsAllDrives=True,
        ).execute()
    except Exception:
        pass

def upload_to_drive(file, user) -> str:
    svc = drive_service()
    folder_id = st.secrets["google_drive"]["attachments_folder_id"]
    unique_name = f"{user}-{int(time.time())}-{file.name}"
    meta = {"name": unique_name, "parents": [folder_id]}
    tmp_path = f"/tmp/{unique_name}"
    with open(tmp_path, "wb") as fh:
        fh.write(file.getvalue())
    media = MediaFileUpload(tmp_path, mimetype=(file.type or "application/octet-stream"), resumable=False)
    created = svc.files().create(body=meta, media_body=media, fields="id", supportsAllDrives=True).execute()
    try: os.remove(tmp_path)
    except Exception: pass
    fid = created["id"]
    make_public_read(fid)  # not required for proxy, but nice if you ever share links
    return fid

def public_view_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def drive_db_last_modified_rfc3339() -> str | None:
    f = get_drive_db_file()
    if not f: return None
    svc = drive_service()
    meta = svc.files().get(fileId=f["id"], fields="modifiedTime").execute()
    return meta.get("modifiedTime")

def local_db_mtime_epoch() -> float | None:
    try: return os.path.getmtime(DB_PATH)
    except FileNotFoundError: return None

def rfc3339_to_epoch(s: str) -> float:
    from datetime import timezone
    s2 = s.replace('Z','')
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if '.' in s2 else "%Y-%m-%dT%H:%M:%S"
    dt = datetime.datetime.strptime(s2, fmt)
    return dt.replace(tzinfo=datetime.timezone.utc).timestamp()

# -------- Proxy: fetch bytes from Drive (cached) --------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_drive_bytes(file_id: str) -> tuple[bytes, str]:
    svc = drive_service()
    meta = svc.files().get(fileId=file_id, fields="mimeType").execute()
    mime = (meta.get("mimeType") or "").lower()
    req = svc.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    buf.seek(0)
    return buf.read(), mime

# ---------------------------- DB ----------------------------
SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

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

CREATE TABLE IF NOT EXISTS images (
  id TEXT PRIMARY KEY,
  entry_id TEXT NOT NULL,
  drive_file_id TEXT NOT NULL,
  mime TEXT,
  FOREIGN KEY(entry_id) REFERENCES entries(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audio (
  id TEXT PRIMARY KEY,
  entry_id TEXT NOT NULL,
  drive_file_id TEXT NOT NULL,
  mime TEXT,
  FOREIGN KEY(entry_id) REFERENCES entries(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS videos (
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
        if not download_db(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            conn.executescript(SCHEMA_SQL)
            conn.commit(); conn.close()
            upload_db(DB_PATH)

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def summarize_text(text, max_sentences=2):
    if not text or not text.strip():
        return ""
    if LsaSummarizer is None:
        return text.split("\n")[0][:300]
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summ = LsaSummarizer()
        sents = summ(parser.document, max_sentences)
        return " ".join(str(s) for s in sents)
    except Exception:
        return text.split("\n")[0][:300]

def save_entry_to_db(user, date, what, meaningful, mood, choice, no_repeat, plans, tags,
                     uploaded_images, uploaded_audio, uploaded_videos):
    entry_id = str(uuid.uuid4())
    tags_str = ", ".join([t.strip() for t in (tags or "").split(",") if t.strip()])
    summary = summarize_text(what)

    conn = get_conn(); c = conn.cursor()
    c.execute("""INSERT INTO entries (id,user,date,what,meaningful,mood,choice,no_repeat,plans,tags,summary)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
              (entry_id, user, date, what, meaningful, int(mood) if mood else None, choice, no_repeat, plans, tags_str, summary))

    for f in (uploaded_images or []):
        fid = upload_to_drive(f, user)
        c.execute("INSERT INTO images (id, entry_id, drive_file_id, mime) VALUES (?,?,?,?)",
                  (str(uuid.uuid4()), entry_id, fid, f.type or ""))

    for f in (uploaded_audio or []):
        fid = upload_to_drive(f, user)
        c.execute("INSERT INTO audio (id, entry_id, drive_file_id, mime) VALUES (?,?,?,?)",
                  (str(uuid.uuid4()), entry_id, fid, f.type or ""))

    for f in (uploaded_videos or []):
        fid = upload_to_drive(f, user)
        c.execute("INSERT INTO videos (id, entry_id, drive_file_id, mime) VALUES (?,?,?,?)",
                  (str(uuid.uuid4()), entry_id, fid, f.type or ""))

    for line in (plans or "").split("\n"):
        t = line.strip()
        if t:
            c.execute("INSERT INTO tasks (id, entry_id, text, is_done) VALUES (?,?,?,0)",
                      (str(uuid.uuid4()), entry_id, t))

    conn.commit(); conn.close(); upload_db(DB_PATH)
    return entry_id

def list_entries_for_user(user, limit=100):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT id, date, what, meaningful, tags, mood FROM entries WHERE user = ? ORDER BY date DESC, created_at DESC LIMIT ?",
        conn, params=(user, limit)
    )
    conn.close(); return df

def load_entry_bundle(user, limit=5):
    conn = get_conn(); c = conn.cursor()
    c.execute("""SELECT id,user,date,what,meaningful,mood,choice,no_repeat,plans,tags,summary,created_at
                 FROM entries WHERE user=? ORDER BY date DESC, created_at DESC LIMIT ?""", (user, limit))
    cols = [d[0] for d in c.description]
    entries = [dict(zip(cols, row)) for row in c.fetchall()]
    for e in entries:
        eid = e["id"]
        c.execute("SELECT id, drive_file_id, mime FROM images WHERE entry_id=?", (eid,))
        e["images"] = [{"id": iid, "file_id": fid, "url": public_view_url(fid), "mime": mime} for (iid, fid, mime) in c.fetchall()]
        c.execute("SELECT id, drive_file_id, mime FROM audio WHERE entry_id=?", (eid,))
        e["audio"] = [{"id": aid, "file_id": fid, "url": public_view_url(fid), "mime": mime} for (aid, fid, mime) in c.fetchall()]
        c.execute("SELECT id, drive_file_id, mime FROM videos WHERE entry_id=?", (eid,))
        e["videos"] = [{"id": vid, "file_id": fid, "url": public_view_url(fid), "mime": mime} for (vid, fid, mime) in c.fetchall()]
        c.execute("SELECT id, text, is_done FROM tasks WHERE entry_id=?", (eid,))
        e["tasks"] = [{"id": tid, "text": t, "is_done": bool(done)} for (tid, t, done) in c.fetchall()]
    conn.close(); return entries

def load_entry_detail(entry_id: str):
    conn = get_conn(); c = conn.cursor()
    c.execute("""SELECT id,user,date,what,meaningful,mood,choice,no_repeat,plans,tags,summary FROM entries WHERE id=?""", (entry_id,))
    row = c.fetchone()
    cols = [d[0] for d in c.description] if row else []
    entry = dict(zip(cols, row)) if row else None
    if entry:
        c.execute("SELECT id, drive_file_id, mime FROM images WHERE entry_id=?", (entry_id,))
        entry["images"] = [{"id": iid, "file_id": fid, "url": public_view_url(fid), "mime": mime} for (iid, fid, mime) in c.fetchall()]
        c.execute("SELECT id, drive_file_id, mime FROM audio WHERE entry_id=?", (entry_id,))
        entry["audio"] = [{"id": aid, "file_id": fid, "url": public_view_url(fid), "mime": mime} for (aid, fid, mime) in c.fetchall()]
        c.execute("SELECT id, drive_file_id, mime FROM videos WHERE entry_id=?", (entry_id,))
        entry["videos"] = [{"id": vid, "file_id": fid, "url": public_view_url(fid), "mime": mime} for (vid, fid, mime) in c.fetchall()]
        c.execute("SELECT id, text, is_done FROM tasks WHERE entry_id=?", (entry_id,))
        entry["tasks"] = [{"id": tid, "text": t, "is_done": bool(done)} for (tid, t, done) in c.fetchall()]
    conn.close(); return entry

def update_entry(entry_id: str, fields: dict):
    keys, vals = [], []
    for k, v in fields.items():
        keys.append(f"{k}=?"); vals.append(v)
    vals.append(entry_id)
    conn = get_conn()
    conn.execute(f"UPDATE entries SET {', '.join(keys)} WHERE id=?", vals)
    conn.commit(); conn.close(); upload_db(DB_PATH)

def delete_image(image_id: str):
    conn = get_conn(); conn.execute("DELETE FROM images WHERE id=?", (image_id,)); conn.commit(); conn.close(); upload_db(DB_PATH)

def delete_audio(audio_id: str):
    conn = get_conn(); conn.execute("DELETE FROM audio WHERE id=?", (audio_id,)); conn.commit(); conn.close(); upload_db(DB_PATH)

def delete_video(video_id: str):
    conn = get_conn(); conn.execute("DELETE FROM videos WHERE id=?", (video_id,)); conn.commit(); conn.close(); upload_db(DB_PATH)

def delete_entry(entry_id: str):
    conn = get_conn(); conn.execute("DELETE FROM entries WHERE id=?", (entry_id,)); conn.commit(); conn.close(); upload_db(DB_PATH)

def add_images(entry_id: str, uploaded_files, user: str):
    if not uploaded_files: return
    conn = get_conn(); c = conn.cursor()
    for f in uploaded_files:
        fid = upload_to_drive(f, user)
        c.execute("INSERT INTO images (id, entry_id, drive_file_id, mime) VALUES (?,?,?,?)",
                  (str(uuid.uuid4()), entry_id, fid, f.type or ""))
    conn.commit(); conn.close(); upload_db(DB_PATH)

def add_audio(entry_id: str, uploaded_files, user: str):
    if not uploaded_files: return
    conn = get_conn(); c = conn.cursor()
    for f in uploaded_files:
        fid = upload_to_drive(f, user)
        c.execute("INSERT INTO audio (id, entry_id, drive_file_id, mime) VALUES (?,?,?,?)",
                  (str(uuid.uuid4()), entry_id, fid, f.type or ""))
    conn.commit(); conn.close(); upload_db(DB_PATH)

def add_videos(entry_id: str, uploaded_files, user: str):
    if not uploaded_files: return
    conn = get_conn(); c = conn.cursor()
    for f in uploaded_files:
        fid = upload_to_drive(f, user)
        c.execute("INSERT INTO videos (id, entry_id, drive_file_id, mime) VALUES (?,?,?,?)",
                  (str(uuid.uuid4()), entry_id, fid, f.type or ""))
    conn.commit(); conn.close(); upload_db(DB_PATH)

def replace_tasks(entry_id: str, new_tasks: list[str]):
    conn = get_conn(); c = conn.cursor()
    c.execute("DELETE FROM tasks WHERE entry_id=?", (entry_id,))
    for t in (new_tasks or []):
        t = t.strip()
        if t:
            c.execute("INSERT INTO tasks (id, entry_id, text, is_done) VALUES (?,?,?,0)",
                      (str(uuid.uuid4()), entry_id, t))
    conn.commit(); conn.close(); upload_db(DB_PATH)

def update_task_done(task_id: str, is_done: bool):
    conn = get_conn()
    conn.execute("UPDATE tasks SET is_done=? WHERE id=?", (1 if is_done else 0, task_id))
    conn.commit(); conn.close(); upload_db(DB_PATH)

def backfill_make_public(user: str | None = None):
    conn = get_conn(); cur = conn.cursor()
    if user:
        cur.execute("""SELECT i.drive_file_id FROM images i JOIN entries e ON e.id=i.entry_id WHERE e.user=?""", (user,))
        img = [r[0] for r in cur.fetchall()]
        cur.execute("""SELECT a.drive_file_id FROM audio a JOIN entries e ON e.id=a.entry_id WHERE e.user=?""", (user,))
        aud = [r[0] for r in cur.fetchall()]
        cur.execute("""SELECT v.drive_file_id FROM videos v JOIN entries e ON e.id=v.entry_id WHERE e.user=?""", (user,))
        vid = [r[0] for r in cur.fetchall()]
    else:
        cur.execute("SELECT drive_file_id FROM images"); img = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT drive_file_id FROM audio");  aud = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT drive_file_id FROM videos"); vid = [r[0] for r in cur.fetchall()]
    conn.close()
    for fid in (img + aud + vid):
        make_public_read(fid)

# ---------------------------- UI ----------------------------
st.set_page_config(page_title="Sanny’s Diary", page_icon="🌀", layout="centered")
st.title("🌀 迷惘但想搞懂的我 / Lost but Learning")
st.caption("SQLite in Google Drive • Proxy media • Images/Audio/Videos split • History=5 • Smart Search • LLM Monthly")

ensure_db()

# Update banner
def rfc3339_to_epoch_safe(s):
    try: return rfc3339_to_epoch(s)
    except Exception: return None

drive_ts = drive_db_last_modified_rfc3339()
local_ts = local_db_mtime_epoch()
if drive_ts:
    drive_epoch = rfc3339_to_epoch_safe(drive_ts)
    if (drive_epoch is not None) and ((local_ts is None) or (drive_epoch > (local_ts + 2))):
        st.warning("🔔 A newer diary database is available in Google Drive.")
        if st.button("🔄 Sync latest diary from Drive"):
            if download_db(DB_PATH): st.success("Synced latest DB from Drive. Reloading…"); st.rerun()

# ===== Sidebar: User + Jump =====
user = st.sidebar.text_input("使用者 / User", value="Sanny")
SECTIONS = [
    "New Entry",
    "Recent Entries",
    "Edit Past Entry",
    "Search Results",
    "Monthly Summary",
    "Export",
    "Settings",
]
section = st.sidebar.radio("🧭 Jump to", SECTIONS, index=0)

# ===== Sections =====
if section == "New Entry":
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
        up_images = st.file_uploader("🖼️ Images", type=["png","jpg","jpeg","gif","bmp","webp"], accept_multiple_files=True)
        up_audio  = st.file_uploader("🔊 Audio",  type=["mp3","wav","ogg","m4a","aac","flac"], accept_multiple_files=True)
        up_videos = st.file_uploader("🎬 Videos", type=["mp4","webm","mov","mkv"], accept_multiple_files=True)
        submitted = st.form_submit_button("提交 / Submit")
        if submitted:
            save_entry_to_db(user, today, what, meaningful, mood, choice, no_repeat, plans, tags,
                             up_images, up_audio, up_videos)
            st.success("已送出！資料與檔案已同步到 Google Drive ✅")

elif section == "Recent Entries":
    st.subheader("📜 歷史紀錄（最近5筆） / Recent Entries")
    entries = load_entry_bundle(user, limit=5)
    if not entries:
        st.info("尚無紀錄。")
    else:
        for e in entries:
            st.markdown(f"**🗓️ {e['date']}** — **Mood:** {e['mood'] if e['mood'] is not None else '-'} /10")
            st.markdown(f"**What:** {e['what'] or ''}")
            if e["meaningful"]:
                st.markdown(f"**Meaningful:** {e['meaningful']}")

            if e.get("images"):
                st.write("**Images:**")
                for a in e["images"]:
                    data, _ = fetch_drive_bytes(a["file_id"])
                    st.image(data, use_container_width=True)

            if e.get("audio"):
                st.write("**Audio:**")
                for a in e["audio"]:
                    data, mime = fetch_drive_bytes(a["file_id"])
                    fmt = "audio/mpeg"
                    mime = (mime or "").lower()
                    if "mp4" in mime or "aac" in mime or "m4a" in mime: fmt = "audio/mp4"
                    elif "wav" in mime: fmt = "audio/wav"
                    elif "ogg" in mime: fmt = "audio/ogg"
                    st.audio(data, format=fmt)

            if e.get("videos"):
                st.write("**Videos:**")
                for a in e["videos"]:
                    data, _ = fetch_drive_bytes(a["file_id"])
                    st.video(data)

            if e["tasks"]:
                st.write("**Tomorrow’s Tasks:**")
                for t in e["tasks"]:
                    new_val = st.checkbox(t["text"], value=t["is_done"], key=f"task-{t['id']}")
                    if new_val != t["is_done"]:
                        update_task_done(t["id"], new_val)

            if st.button("🗑️ 刪除這筆 / Delete this entry", key=f"del-entry-{e['id']}"):
                delete_entry(e["id"]); st.success("Entry deleted."); st.rerun()

elif section == "Edit Past Entry":
    st.subheader("✏️ 編輯過去日記 / Edit Past Entry")
    def entry_label(r): return f"{r['date']} | {str(r['what'] or '')[:40]}"
    opts = list_entries_for_user(user, limit=200)
    if opts.empty:
        st.info("沒有可編輯的紀錄。")
    else:
        opts["label"] = opts.apply(entry_label, axis=1)
        chosen = st.selectbox("選擇要編輯的日記 / Select entry", opts["label"].tolist())
        if chosen:
            sel_id = opts.loc[opts["label"] == chosen, "id"].iloc[0]
            entry = load_entry_detail(sel_id)
            if entry:
                with st.form("edit_entry_form", clear_on_submit=False):
                    new_date = st.text_input("日期 / Date (YYYY-MM-DD)", entry["date"])
                    new_what = st.text_area("What did you do today?", entry["what"] or "", height=140)
                    new_meaningful = st.text_area("Meaningful event", entry["meaningful"] or "")
                    new_mood = st.slider("Mood (1-10)", 1, 10, int(entry["mood"] or 5))
                    new_choice = st.text_area("Was it your choice?", entry["choice"] or "")
                    new_no_repeat = st.text_area("What you wouldn't repeat", entry["no_repeat"] or "")
                    existing_tasks = "\n".join([t["text"] for t in entry["tasks"]]) if entry["tasks"] else ""
                    new_plans = st.text_area("Plans for tomorrow (one per line)", existing_tasks)
                    new_tags = st.text_input("Tags (comma-separated)", entry["tags"] or "")
                    add_imgs = st.file_uploader("新增圖片 / Add images", type=["png","jpg","jpeg","gif","bmp","webp"], accept_multiple_files=True)
                    add_auds = st.file_uploader("新增音訊 / Add audio", type=["mp3","wav","ogg","m4a","aac","flac"], accept_multiple_files=True)
                    add_vids = st.file_uploader("新增影片 / Add videos", type=["mp4","webm","mov","mkv"], accept_multiple_files=True)
                    submitted_edit = st.form_submit_button("儲存變更 / Save changes")

                if entry.get("images"):
                    st.write("現有圖片 / Existing images:")
                    for a in entry["images"]:
                        col1, col2 = st.columns([4,1])
                        with col1:
                            data, _ = fetch_drive_bytes(a["file_id"])
                            st.image(data,use_container_width=True)
                        with col2:
                            if st.button("刪除 / Delete", key=f"del-img-{a['id']}"):
                                delete_image(a["id"]); st.rerun()

                if entry.get("audio"):
                    st.write("現有音訊 / Existing audio:")
                    for a in entry["audio"]:
                        col1, col2 = st.columns([4,1])
                        with col1:
                            data, mime = fetch_drive_bytes(a["file_id"])
                            fmt = "audio/mpeg"
                            mime = (mime or "").lower()
                            if "mp4" in mime or "aac" in mime or "m4a" in mime: fmt = "audio/mp4"
                            elif "wav" in mime: fmt = "audio/wav"
                            elif "ogg" in mime: fmt = "audio/ogg"
                            st.audio(data, format=fmt)
                        with col2:
                            if st.button("刪除 / Delete", key=f"del-aud-{a['id']}"):
                                delete_audio(a["id"]); st.rerun()

                if entry.get("videos"):
                    st.write("現有影片 / Existing videos:")
                    for a in entry["videos"]:
                        col1, col2 = st.columns([4,1])
                        with col1:
                            data, _ = fetch_drive_bytes(a["file_id"])
                            st.video(data)
                        with col2:
                            if st.button("刪除 / Delete", key=f"del-vid-{a['id']}"):
                                delete_video(a["id"]); st.rerun()

                if submitted_edit:
                    summary = summarize_text(new_what)
                    update_entry(sel_id, {
                        "date": new_date, "what": new_what, "meaningful": new_meaningful,
                        "mood": int(new_mood), "choice": new_choice, "no_repeat": new_no_repeat,
                        "plans": new_plans, "tags": ", ".join([t.strip() for t in (new_tags or '').split(',') if t.strip()]),
                        "summary": summary
                    })
                    replace_tasks(sel_id, new_plans.split("\n") if new_plans else [])
                    add_images(sel_id, add_imgs, user)
                    add_audio(sel_id, add_auds, user)
                    add_videos(sel_id, add_vids, user)
                    st.success("已更新！DB 已同步到 Google Drive ✅"); st.rerun()

elif section == "Search Results":
    st.subheader("🔎 搜尋結果 / Search Results")
    q = st.text_input("Keywords (space-separated)", key="q")
    tag_query = st.text_input("Filter tags (comma-separated)", key="tags_q")
    mood_min, mood_max = st.slider("Mood range", 1, 10, (1, 10), key="mood_q")
    c1, c2 = st.columns(2)
    date_from = c1.date_input("From", value=None, key="from_q")
    date_to   = c2.date_input("To", value=None, key="to_q")

    conn = get_conn()
    df_all = pd.read_sql_query("SELECT * FROM entries WHERE user = ? ORDER BY date DESC, created_at DESC", conn, params=(user,))
    conn.close()
    if df_all.empty:
        st.info("無資料可搜尋。")
    else:
        def score_row(row, q_tokens, tag_tokens, mood_range, date_from, date_to):
            score = 0
            text_fields = " ".join([
                str(row.get("what","")), str(row.get("meaningful","")),
                str(row.get("plans","")), str(row.get("tags",""))
            ]).lower()
            for tok in q_tokens:
                if tok in text_fields: score += 3
            for t in tag_tokens:
                if t in (row.get("tags","").lower()): score += 2
            mood = row.get("mood", None)
            if mood is not None:
                try:
                    mood = int(mood)
                    if mood_range and (mood < mood_range[0] or mood > mood_range[1]):
                        return -1
                except Exception:
                    pass
            try:
                d = datetime.datetime.strptime(row.get("date",""), "%Y-%m-%d").date()
                if date_from and d < date_from: return -1
                if date_to and d > date_to: return -1
            except Exception:
                pass
            return score

        q_tokens = [t.strip().lower() for t in (q or "").split() if t.strip()]
        tag_tokens = [t.strip().lower() for t in (tag_query or "").split(",") if t.strip()]
        d_from = date_from if isinstance(date_from, datetime.date) else None
        d_to = date_to if isinstance(date_to, datetime.date) else None
        df_all["__score"] = df_all.apply(lambda r: score_row(r, q_tokens, tag_tokens, (mood_min, mood_max), d_from, d_to), axis=1)
        res = df_all[df_all["__score"] >= 0].sort_values(["__score","date"], ascending=[False, False]).head(50)
        if res.empty:
            st.info("找不到符合的結果。")
        else:
            for _, r in res.iterrows():
                st.markdown(f"**{r['date']}** — Mood: {r['mood'] if pd.notnull(r['mood']) else '-'}")
                st.write((r["what"] or "")[:280])
                if r.get("tags"): st.caption(f"Tags: {r['tags']}")
                st.markdown("---")

elif section == "Monthly Summary":
    st.subheader("🗓️ 每月總結 / Monthly Summary")
    now = datetime.date.today()
    coly, colm = st.columns(2)
    year = coly.number_input("Year", min_value=2000, max_value=2100, value=now.year, step=1)
    month = colm.number_input("Month", min_value=1, max_value=12, value=now.month, step=1)
    if st.button("Generate Monthly Summary"):
        def llm_monthly_summary(user: str, year: int, month: int) -> str:
            conn = get_conn()
            start = datetime.date(year, month, 1)
            end = (datetime.date(year+1,1,1)-datetime.timedelta(days=1)) if month==12 else (datetime.date(year,month+1,1)-datetime.timedelta(days=1))
            df = pd.read_sql_query("""
                SELECT date, what, meaningful, mood, tags FROM entries
                WHERE user = ? AND date BETWEEN ? AND ?
                ORDER BY date ASC
            """, conn, params=(user, start.isoformat(), end.isoformat()))
            conn.close()
            if df.empty: return "No entries this month."
            lines = []
            for _, r in df.iterrows():
                lines.append(f"{r['date']}: {str(r['what'] or '')} | Meaningful: {str(r['meaningful'] or '')} | Mood: {str(r['mood'] or '')} | Tags: {str(r['tags'] or '')}")
            digest = "\n".join(lines)
            if OPENAI_AVAILABLE:
                try:
                    prompt = ("Summarize the following daily diary lines into a monthly reflection. "
                              "Highlight patterns, wins, struggles, and 3 actionable suggestions for next month. "
                              "Keep it under 180 words.\n\n" + digest)
                    resp = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role":"system","content":"You are a helpful, concise coach."},
                                  {"role":"user","content": prompt}],
                        temperature=0.5,
                    )
                    return resp.choices[0].message["content"].strip()
                except Exception:
                    pass
            if LsaSummarizer is None: return digest[:1000]
            try:
                parser = PlaintextParser.from_string(digest, Tokenizer("english"))
                summ = LsaSummarizer()
                sents = summ(parser.document, 6)
                return " ".join(str(s) for s in sents)
            except Exception:
                return digest[:1000]

        with st.spinner("Summarizing…"):
            summary_text = llm_monthly_summary(user, int(year), int(month))
        st.success("Done!")
        st.write(summary_text)

elif section == "Export":
    st.subheader("📤 匯出 / Export")
    exists = bool(list_entries_for_user(user, limit=1).shape[0])
    if exists:
        conn = get_conn()
        df = pd.read_sql_query("SELECT * FROM entries WHERE user = ? ORDER BY date DESC, created_at DESC", conn, params=(user,))
        conn.close()
        st.download_button("下載 CSV (entries)", df.to_csv(index=False).encode("utf-8-sig"), file_name="entries.csv", mime="text/csv")
    else:
        st.info("沒有資料可以匯出。")

elif section == "Settings":
    st.subheader("⚙️ Settings")
    st.write("資料永久保存：DB 與附件都在 Google Drive（媒體以代理模式顯示）。")
    if st.button("Make my attachments public (anyone with link)"):
        backfill_make_public(user); st.success("Done.")
