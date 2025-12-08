import os, io, re, uuid, time, sqlite3, datetime
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
client = None
try:
    from openai import OpenAI
    api_key = None
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        api_key = st.secrets["openai"]["api_key"]
    elif os.getenv("OPENAI_API_KEY"):
        api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
        OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
    client = None

# --- Google Drive client (non-resumable uploads for reliability) ---
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

DB_PATH = "journal.sqlite"

# ---------- Google Drive ----------
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
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    fid = created["id"]
    make_public_read(fid)  # not required for proxy, but nice if you ever share links
    return fid

def public_view_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def drive_db_last_modified_rfc3339() -> str | None:
    f = get_drive_db_file()
    if not f:
        return None
    svc = drive_service()
    meta = svc.files().get(fileId=f["id"], fields="modifiedTime").execute()
    return meta.get("modifiedTime")

def local_db_mtime_epoch() -> float | None:
    try:
        return os.path.getmtime(DB_PATH)
    except FileNotFoundError:
        return None

def rfc3339_to_epoch(s: str) -> float:
    from datetime import timezone
    s2 = s.replace("Z", "")
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in s2 else "%Y-%m-%dT%H:%M:%S"
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

# -------- Drive metadata helper (cached) --------
@st.cache_data(show_spinner=False, ttl=3600)
def get_drive_file_info(file_id: str) -> dict:
    """Get file metadata including original filename from Drive"""
    svc = drive_service()
    try:
        meta = svc.files().get(fileId=file_id, fields="name,mimeType,size").execute()
        return {
            "name": meta.get("name", "unknown_file"),
            "mime": meta.get("mimeType", "").lower(),
            "size": int(meta.get("size", 0)),
        }
    except Exception:
        return {"name": "unknown_file", "mime": "", "size": 0}

# ---------- DB ----------
SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS entries (
  id TEXT PRIMARY KEY,
  user TEXT NOT NULL,
  date TEXT NOT NULL,
  what TEXT,
  meaningful TEXT,
  mood INTEGER,
  mood_note TEXT,
  story TEXT,
  choice TEXT,
  no_repeat TEXT,
  plans_today TEXT,
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
  original_name TEXT,
  FOREIGN KEY(entry_id) REFERENCES entries(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audio (
  id TEXT PRIMARY KEY,
  entry_id TEXT NOT NULL,
  drive_file_id TEXT NOT NULL,
  mime TEXT,
  original_name TEXT,
  FOREIGN KEY(entry_id) REFERENCES entries(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS videos (
  id TEXT PRIMARY KEY,
  entry_id TEXT NOT NULL,
  drive_file_id TEXT NOT NULL,
  mime TEXT,
  original_name TEXT,
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
            conn.commit()
            conn.close()
            upload_db(DB_PATH)
    else:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Ensure original_name exists on all media tables
        for tbl in ("images", "audio", "videos"):
            cursor.execute(f"PRAGMA table_info({tbl})")
            cols = [c[1] for c in cursor.fetchall()]
            if "original_name" not in cols:
                cursor.execute(f"ALTER TABLE {tbl} ADD COLUMN original_name TEXT")
        # Ensure new entry columns exist
        cursor.execute("PRAGMA table_info(entries)")
        cols = [c[1] for c in cursor.fetchall()]
        if "mood_note" not in cols:
            cursor.execute("ALTER TABLE entries ADD COLUMN mood_note TEXT")
        if "story" not in cols:
            cursor.execute("ALTER TABLE entries ADD COLUMN story TEXT")
        if "plans_today" not in cols:
            cursor.execute("ALTER TABLE entries ADD COLUMN plans_today TEXT")
        conn.commit()
        conn.close()

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

# ---------- Filename fixes ----------
def strip_upload_prefix(name: str) -> str:
    """
    Uploaded files are saved as `user-<epoch>-original.ext`.
    Remove the leading `user-<epoch>-` if present.
    """
    if not name:
        return name
    return re.sub(r"^[^-]+-\d{9,}-", "", name)

def resolve_display_name(file_id: str, original_name: str | None, fallback_label: str) -> str:
    """
    Prefer DB's original_name; otherwise use Drive's filename (stripped).
    """
    if original_name and original_name.strip():
        return original_name.strip()
    info = get_drive_file_info(file_id)
    drive_name = strip_upload_prefix(info.get("name", "")) if info else ""
    return drive_name or fallback_label

# ---------- Generic media helpers (merge of images/audio/videos) ----------
MEDIA_MAP = {
    "images": {"table": "images"},
    "audio":  {"table": "audio"},
    "videos": {"table": "videos"},
}

def add_media(entry_id: str, uploaded_files, user: str, kind: str):
    """Insert uploaded files of a media kind into its table."""
    if not uploaded_files:
        return
    tbl = MEDIA_MAP[kind]["table"]
    conn = get_conn(); cur = conn.cursor()
    for f in uploaded_files:
        fid = upload_to_drive(f, user)
        cur.execute(
            f"INSERT INTO {tbl} (id, entry_id, drive_file_id, mime, original_name) VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), entry_id, fid, f.type or "", f.name),
        )
    conn.commit(); conn.close(); upload_db(DB_PATH)

def delete_media(media_id: str, kind: str):
    """Delete a single media row by id for a given kind."""
    tbl = MEDIA_MAP[kind]["table"]
    conn = get_conn()
    conn.execute(f"DELETE FROM {tbl} WHERE id=?", (media_id,))
    conn.commit(); conn.close(); upload_db(DB_PATH)

def list_media(entry_id: str, kind: str):
    """Return [{id,file_id,url,mime,original_name}] for a given entry/kind."""
    tbl = MEDIA_MAP[kind]["table"]
    conn = get_conn(); cur = conn.cursor()
    cur.execute(f"SELECT id, drive_file_id, mime, original_name FROM {tbl} WHERE entry_id=?", (entry_id,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for (mid, fid, mime, name) in rows:
        disp = resolve_display_name(fid, name, kind[:-1])  # "images"->"image"
        out.append({
            "id": mid,
            "file_id": fid,
            "url": public_view_url(fid),
            "mime": mime,
            "original_name": disp
        })
    return out

# ---------- Entries CRUD ----------
def save_entry_to_db(
    user,
    date,
    what,
    meaningful,
    mood,
    mood_note,
    story,
    choice,
    no_repeat,
    plans_today,
    plans_tomorrow,
    tags,
    uploaded_images,
    uploaded_audio,
    uploaded_videos,
):
    entry_id = str(uuid.uuid4())
    tags_str = ", ".join([t.strip() for t in (tags or "").split(",") if t.strip()])
    summary = summarize_text(what)

    conn = get_conn(); c = conn.cursor()
    c.execute(
        """INSERT INTO entries
           (id,user,date,what,meaningful,mood,mood_note,story,choice,no_repeat,plans_today,plans,tags,summary)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            entry_id,
            user,
            date,
            what,
            meaningful,
            int(mood) if mood else None,
            mood_note,
            story,
            choice,
            no_repeat,
            plans_today,
            plans_tomorrow,
            tags_str,
            summary,
        ),
    )
    conn.commit(); conn.close()

    # add media (generic)
    add_media(entry_id, uploaded_images, user, "images")
    add_media(entry_id, uploaded_audio,  user, "audio")
    add_media(entry_id, uploaded_videos, user, "videos")

    upload_db(DB_PATH)
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
    c.execute(
        """SELECT id,user,date,what,meaningful,mood,mood_note,story,choice,no_repeat,plans_today,plans,tags,summary,created_at
           FROM entries WHERE user=? ORDER BY date DESC, created_at DESC LIMIT ?""",
        (user, limit),
    )
    cols = [d[0] for d in c.description]
    entries = [dict(zip(cols, row)) for row in c.fetchall()]
    conn.close()

    for e in entries:
        eid = e["id"]
        e["images"] = list_media(eid, "images")
        e["audio"]  = list_media(eid, "audio")
        e["videos"] = list_media(eid, "videos")

        # tasks
        conn2 = get_conn(); cur2 = conn2.cursor()
        cur2.execute("SELECT id, text, is_done FROM tasks WHERE entry_id=?", (eid,))
        e["tasks"] = [{"id": tid, "text": t, "is_done": bool(done)} for (tid, t, done) in cur2.fetchall()]
        conn2.close()
    return entries

def load_entry_detail(entry_id: str):
    conn = get_conn(); c = conn.cursor()
    c.execute(
        """SELECT id,user,date,what,meaningful,mood,mood_note,story,choice,no_repeat,plans_today,plans,tags,summary
           FROM entries WHERE id=?""",
        (entry_id,),
    )
    row = c.fetchone()
    cols = [d[0] for d in c.description] if row else []
    entry = dict(zip(cols, row)) if row else None
    if entry:
        entry["images"] = list_media(entry_id, "images")
        entry["audio"]  = list_media(entry_id, "audio")
        entry["videos"] = list_media(entry_id, "videos")
        c.execute("SELECT id, text, is_done FROM tasks WHERE entry_id=?", (entry_id,))
        entry["tasks"] = [{"id": tid, "text": t, "is_done": bool(done)} for (tid, t, done) in c.fetchall()]
    conn.close(); return entry

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

def delete_entry(entry_id: str):
    conn = get_conn(); conn.execute("DELETE FROM entries WHERE id=?", (entry_id,)); conn.commit(); conn.close(); upload_db(DB_PATH)

# ---------- Utilities ----------
def rfc3339_to_epoch_safe(s):
    try:
        return rfc3339_to_epoch(s)
    except Exception:
        return None

def format_file_size(size_bytes):
    """Convert bytes to human readable format"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def create_download_button(file_id: str, original_name: str, media_type: str, key_suffix: str):
    """Create a download button for media files"""
    try:
        data, _ = fetch_drive_bytes(file_id)
        file_info = get_drive_file_info(file_id)
        file_size = format_file_size(file_info.get("size", len(data)))
        st.download_button(
            label=f"📥 Download ({file_size})",
            data=data,
            file_name=original_name,
            mime=file_info.get("mime", "application/octet-stream"),
            key=f"download_{media_type}_{key_suffix}"
        )
    except Exception as e:
        st.error(f"Error creating download button: {str(e)}")

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

def backfill_original_names(user: str | None = None):
    """
    Populate original_name in images/audio/videos where it's NULL or empty,
    using Drive metadata (with upload prefix stripped).
    """
    conn = get_conn(); cur = conn.cursor()
    tables = ["images", "audio", "videos"]
    for tbl in tables:
        if user:
            cur.execute(f"""
                SELECT {tbl}.id, {tbl}.drive_file_id
                FROM {tbl} JOIN entries e ON e.id = {tbl}.entry_id
                WHERE ( {tbl}.original_name IS NULL OR TRIM({tbl}.original_name)='' )
                  AND e.user = ?
            """, (user,))
        else:
            cur.execute(f"""
                SELECT i
