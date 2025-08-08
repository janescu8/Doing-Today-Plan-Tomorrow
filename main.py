# main.py
import os, io, uuid, time, sqlite3, datetime
import streamlit as st
import pandas as pd

# --- Optional summarizer (no external LLM API) ---
try:
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
except Exception:
    LsaSummarizer = None

# --- Google Drive client (non-resumable uploads for Streamlit Cloud reliability) ---
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

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
    # Works for My Drive and Shared Drives
    q = f"name = '{name}' and '{folder_id}' in parents and trashed = false"
    resp = svc.files().list(
        q=q,
        fields="files(id, name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        corpora="allDrives",
        spaces="drive",
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

    # SIMPLE upload (non-resumable) – avoids ResumableUploadError
    media = MediaFileUpload(local_path, mimetype="application/octet-stream", resumable=False)
    file_meta = {"name": fname, "parents": [folder_id]}

    if f:
        svc.files().update(
            fileId=f["id"],
            media_body=media,
            supportsAllDrives=True,
        ).execute()
    else:
        svc.files().create(
            body=file_meta,
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()

def make_public_read(file_id: str):
    """Grant 'anyone with the link' read access so media renders in the browser."""
    svc = drive_service()
    try:
        svc.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
            supportsAllDrives=True,
        ).execute()
    except Exception:
        # Ignore if permission already exists or org policy blocks it
        pass

def upload_attachment(file, user) -> str:
    """Upload a file to Drive attachments folder, return fileId (and make it public-read)."""
    svc = drive_service()
    folder_id = st.secrets["google_drive"]["attachments_folder_id"]
    unique_name = f"{user}-{int(time.time())}-{file.name}"
    file_meta = {"name": unique_name, "parents": [folder_id]}

    # Save to temp then simple upload (reliable)
    tmp_path = f"/tmp/{unique_name}"
    with open(tmp_path, "wb") as fh:
        fh.write(file.getvalue())

    media = MediaFileUpload(tmp_path, mimetype=(file.type or "application/octet-stream"), resumable=False)
    created = svc.files().create(
        body=file_meta,
        media_body=media,
        fields="id",
        supportsAllDrives=True,
    ).execute()

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    file_id = created["id"]
    make_public_read(file_id)   # ensure browser can load it
    return file_id

def public_view_url(file_id: str) -> str:
    # “export=download” avoids some Drive preview quirks and streams cleanly
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def drive_db_last_modified_rfc3339() -> str | None:
    f = get_drive_db_file()
    if not f:
        return None
    svc = drive_service()
    meta = svc.files().get(fileId=f["id"], fields="modifiedTime").execute()
    return meta.get("modifiedTime")  # e.g. '2025-08-08T06:22:36.000Z'

def local_db_mtime_epoch() -> float | None:
    try:
        return os.path.getmtime(DB_PATH)
    except FileNotFoundError:
        return None

def rfc3339_to_epoch(s: str) -> float:
    from datetime import timezone
    # 'YYYY-MM-DDTHH:MM:SS.sssZ' or without ms
    s2 = s.replace('Z','')
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if '.' in s2 else "%Y-%m-%dT%H:%M:%S"
    dt = datetime.datetime.strptime(s2, fmt)
    return dt.replace(tzinfo=datetime.timezone.utc).timestamp()

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

def list_entries_for_user(user, limit=100):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT id, date, what FROM entries WHERE user = ? ORDER BY date DESC, created_at DESC LIMIT ?",
        conn, params=(user, limit)
    )
    conn.close()
    return df

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

def load_entry_detail(entry_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""SELECT id,user,date,what,meaningful,mood,choice,no_repeat,plans,tags,summary
                 FROM entries WHERE id=?""", (entry_id,))
    row = c.fetchone()
    cols = [d[0] for d in c.description] if row else []
    entry = dict(zip(cols, row)) if row else None
    if entry:
        c.execute("SELECT id, drive_file_id, mime FROM attachments WHERE entry_id=?", (entry_id,))
        entry_attachments = [{"id": aid, "file_id": fid, "url": public_view_url(fid), "mime": mime} for (aid, fid, mime) in c.fetchall()]
        c.execute("SELECT id, text, is_done FROM tasks WHERE entry_id=?", (entry_id,))
        entry_tasks = [{"id": tid, "text": t, "is_done": bool(done)} for (tid, t, done) in c.fetchall()]
        entry["attachments"] = entry_attachments
        entry["tasks"] = entry_tasks
    conn.close()
    return entry

def update_entry(entry_id: str, fields: dict):
    keys = []
    vals = []
    for k, v in fields.items():
        keys.append(f"{k}=?")
        vals.append(v)
    vals.append(entry_id)
    conn = get_conn()
    conn.execute(f"UPDATE entries SET {', '.join(keys)} WHERE id=?", vals)
    conn.commit()
    conn.close()
    upload_db(DB_PATH)

def delete_attachment(attachment_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM attachments WHERE id=?", (attachment_id,))
    conn.commit()
    conn.close()
    upload_db(DB_PATH)

def add_attachments(entry_id: str, uploaded_files, user: str):
    if not uploaded_files:
        return
    conn = get_conn()
    c = conn.cursor()
    for f in uploaded_files:
        fid = upload_attachment(f, user)
        c.execute("INSERT INTO attachments (id, entry_id, drive_file_id, mime) VALUES (?,?,?,?)",
                  (str(uuid.uuid4()), entry_id, fid, f.type or ""))
    conn.commit()
    conn.close()
    upload_db(DB_PATH)

def replace_tasks(entry_id: str, new_tasks: list[str]):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM tasks WHERE entry_id=?", (entry_id,))
    for t in new_tasks:
        if t.strip():
            c.execute("INSERT INTO tasks (id, entry_id, text, is_done) VALUES (?,?,?,0)",
                      (str(uuid.uuid4()), entry_id, t.strip()))
    conn.commit()
    conn.close()
    upload_db(DB_PATH)

def update_task_done(task_id: str, is_done: bool):
    conn = get_conn()
    conn.execute("UPDATE tasks SET is_done=? WHERE id=?", (1 if is_done else 0, task_id))
    conn.commit()
    conn.close()
    upload_db(DB_PATH)

def backfill_make_attachments_public(user: str | None = None):
    """Give public-read to all existing attachment files (useful after switching policy)."""
    conn = get_conn()
    cur = conn.cursor()
    if user:
        cur.execute("""
            SELECT a.drive_file_id FROM attachments a
            JOIN entries e ON e.id = a.entry_id
            WHERE e.user = ?
        """, (user,))
    else:
        cur.execute("SELECT drive_file_id FROM attachments")
    rows = cur.fetchall()
    conn.close()
    for (fid,) in rows:
        make_public_read(fid)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Sanny’s Diary", page_icon="🌀", layout="centered")
st.title("🌀 迷惘但想搞懂的我 / Lost but Learning")
st.caption("SQLite stored in Google Drive • Attachments in Drive (public link) • Summaries via sumy")

ensure_db()

# === UPDATE AVAILABLE BANNER ===
def rfc3339_to_epoch_safe(s):
    try:
        return rfc3339_to_epoch(s)
    except Exception:
        return None

drive_ts = drive_db_last_modified_rfc3339()
local_ts = local_db_mtime_epoch()
if drive_ts:
    drive_epoch = rfc3339_to_epoch_safe(drive_ts)
    if (drive_epoch is not None) and ((local_ts is None) or (drive_epoch > (local_ts + 2))):
        st.warning("🔔 A newer diary database is available in Google Drive.")
        if st.button("🔄 Sync latest diary from Drive"):
            if download_db(DB_PATH):
                st.success("Synced latest DB from Drive. Reloading…")
                st.rerun()

# Sidebar utilities
user = st.sidebar.text_input("使用者 / User", value="Sanny")
st.sidebar.info("資料永久保存：DB 與附件都存放在 Google Drive（附件為公開連結）。")

with st.sidebar.expander("🔧 Fix attachments permissions"):
    if st.button("Make my attachments public (anyone with link)"):
        backfill_make_attachments_public(user)
        st.success("Done. Refreshing…")
        st.rerun()

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
    uploaded_files = st.file_uploader(
        "📎 附加檔案 / Attachments (image or audio)",
        type=["png","jpg","jpeg","gif","bmp","webp","mp3","wav","ogg","m4a","aac","flac"],
        accept_multiple_files=True
    )
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
                mime = (a.get("mime") or "").lower()
                url = a["url"]
                ext = os.path.splitext(url.split("?")[0])[1].lower()
                is_image = mime.startswith("image/") or ext in [".png",".jpg",".jpeg",".gif",".bmp",".webp"]
                is_audio = mime.startswith("audio/") or ext in [".mp3",".wav",".ogg",".m4a",".aac",".flac"]
                if is_image:
                    st.image(url, width=240)
                elif is_audio:
                    st.audio(url)
                else:
                    st.write(url)

        # Tasks
        if e["tasks"]:
            st.write("**Tomorrow’s Tasks:**")
            for t in e["tasks"]:
                new_val = st.checkbox(t["text"], value=t["is_done"], key=f"task-{t['id']}")
                if new_val != t["is_done"]:
                    update_task_done(t["id"], new_val)

st.divider()
st.subheader("✏️ 編輯過去日記 / Edit Past Entry")

def entry_label(r):
    return f"{r['date']} | {str(r['what'] or '')[:40]}"

options_df = list_entries_for_user(user, limit=200)
if options_df.empty:
    st.info("沒有可編輯的紀錄。")
else:
    options_df["label"] = options_df.apply(entry_label, axis=1)
    selected_label = st.selectbox("選擇要編輯的日記 / Select entry", options_df["label"].tolist())
    if selected_label:
        sel_id = options_df.loc[options_df["label"] == selected_label, "id"].iloc[0]
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
                add_files = st.file_uploader("新增附件 / Add attachments (image/audio)",
                                              type=["png","jpg","jpeg","gif","bmp","webp","mp3","wav","ogg","m4a","aac","flac"],
                                              accept_multiple_files=True)
                submitted_edit = st.form_submit_button("儲存變更 / Save changes")

            if entry.get("attachments"):
                st.write("現有附件 / Existing attachments:")
                for a in entry["attachments"]:
                    col1, col2 = st.columns([4,1])
                    with col1:
                        mime = (a.get("mime") or "").lower()
                        if mime.startswith("image/"):
                            st.image(a["url"], width=220)
                        elif mime.startswith("audio/"):
                            st.audio(a["url"])
                        else:
                            st.write(a["url"])
                    with col2:
                        if st.button("刪除 / Delete", key=f"del-{a['id']}"):
                            delete_attachment(a["id"])
                            st.rerun()

            if submitted_edit:
                summary = summarize_text(new_what)
                update_entry(sel_id, {
                    "date": new_date,
                    "what": new_what,
                    "meaningful": new_meaningful,
                    "mood": int(new_mood),
                    "choice": new_choice,
                    "no_repeat": new_no_repeat,
                    "plans": new_plans,
                    "tags": ", ".join([t.strip() for t in (new_tags or '').split(',') if t.strip()]),
                    "summary": summary
                })
                replace_tasks(sel_id, new_plans.split("\n") if new_plans else [])
                add_attachments(sel_id, add_files, user)
                st.success("已更新！DB 已同步到 Google Drive ✅")
                st.rerun()

st.divider()
st.subheader("📤 匯出 / Export")
entries_exist = bool(list_entries_for_user(user, limit=1).shape[0])
if entries_exist:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM entries WHERE user = ? ORDER BY date DESC, created_at DESC", conn, params=(user,))
    conn.close()
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下載 CSV (entries)", csv, file_name="entries.csv", mime="text/csv")

st.caption("Note: Attachments are made public-read on upload so the browser can display them. If your org forbids public sharing, we can switch to a private proxy mode.")
