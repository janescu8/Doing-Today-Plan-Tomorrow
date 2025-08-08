import os
import streamlit as st
import datetime
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit.components.v1 as components
from wordcloud import WordCloud
from io import BytesIO
# Summarization imports from sumy
try:
    # Use an extractive summarizer as a lightweight alternative to LLMs
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
except Exception:
    # If sumy isn't available, define dummy summarizer
    LsaSummarizer = None


"""
Enhanced diary application for Sanny.

This version builds upon the basic Streamlit journal app by adding
additional features across several categories:
  * Visual analytics – weekly mood bar chart and tag filtering.
  * Content organization – tag filter to view entries by tag.
  * Reminders & notifications – an "On this day" section shows past entries
    from the same date in previous years.
  * Analysis & AI integration – a simple sentiment analysis using
    predefined positive and negative word lists.
  * Attachments & media – upload images or audio files along with entries.
  * Habit & task management – display tomorrow’s plans as a checklist.
  * Data export & integration – export entries as CSV or JSON.

Column names in the Google Sheet must be English: [User, Date, What did you do today?,
Meaningful Event, Mood, Was it your choice?, What you wouldn’t repeat,
Plans for tomorrow, Tags, Attachments].  For backwards‑compatibility,
entries without Attachments will still load correctly.
"""

## Google Sheets Setup
# Global variable to store the sheet header.  This is used later when
# determining column indices for updating task statuses and summaries.
HEADER_ROW: list = []
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(st.secrets["google_auth"], scopes=scope)
client = gspread.authorize(creds)
sheet = client.open("journal_export").sheet1

# Ensure the Google Sheet has the required extra columns for task statuses and summaries.
# Fetch the header row and append missing headers if necessary.  This prevents
# runtime errors when reading/writing these fields later on.
try:
    header_row = sheet.row_values(1)
    # Expose header for later use in updating task statuses
    # Update the global HEADER_ROW in outer scope
    HEADER_ROW = header_row.copy()
    updated = False
    # Add Task Statuses column if absent
    if 'Task Statuses' not in header_row:
        header_row.append('Task Statuses')
        updated = True
    # Add Summary column if absent
    if 'Summary' not in header_row:
        header_row.append('Summary')
        updated = True
    if updated:
        # Update only the header row; gspread expects a 2D list
        sheet.update('A1', [header_row])
except Exception:
    # If header retrieval fails (e.g. no sheet), silently proceed
    pass

## Utility functions
def render_multiline(text: str) -> str:
    """
    Replace newline characters with HTML `<br>` tags so that
    multi‑line journal entries retain their original formatting
    when displayed with `st.markdown` (with `unsafe_allow_html=True`).
    """
    if not isinstance(text, str):
        text = str(text)
    return text.replace('\n', '<br>')


def sentiment_score(text: str) -> int:
    """
    Compute a simple sentiment score based on positive/negative word lists.
    Returns +1 for positive, -1 for negative, 0 for neutral.
    """
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    positives = {"good", "great", "happy", "satisfied", "amazing", "nice", "wonderful", "love", "enjoy", "success", "win"}
    negatives = {"bad", "sad", "unhappy", "terrible", "awful", "hate", "angry", "frustrated", "fail", "loss", "tired"}
    pos_count = sum(word in text_lower for word in positives)
    neg_count = sum(word in text_lower for word in negatives)
    if pos_count > neg_count:
        return 1
    elif neg_count > pos_count:
        return -1
    return 0


def save_attachments(user: str, date: str, uploaded_files) -> str:
    """
    Save uploaded files to a user‑specific folder and return a semicolon
    separated list of file paths.  If no files are uploaded, return an
    empty string.  Only image and audio files are supported for display.
    """
    if not uploaded_files:
        return ""
    upload_dir = os.path.join("uploads", user)
    os.makedirs(upload_dir, exist_ok=True)
    paths = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    for file in uploaded_files:
        filename = f"{date}_{timestamp}_{file.name}"
        save_path = os.path.join(upload_dir, filename)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        paths.append(save_path)
    return ";".join(paths)


def generate_summary(text: str, sentences: int = 2) -> str:
    """
    Generate a short summary from the provided text.  This uses the LSA summarizer
    from the sumy library if available; otherwise it falls back to extracting the
    first few sentences or returning the original text.  Summaries help
    condense journal entries for quick review.

    Parameters
    ----------
    text : str
        The full text to summarize.
    sentences : int, optional
        The maximum number of sentences to include in the summary.

    Returns
    -------
    str
        A concise summary of the text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    # Use the extractive summarizer when available
    if LsaSummarizer is not None:
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary_sentences = summarizer(parser.document, sentences)
            # Join the sentences into a single string
            return " ".join(str(sent).strip() for sent in summary_sentences)
        except Exception:
            pass
    # Fallback: return the first few sentences split by common delimiters
    import re
    sentence_list = re.split(r'[\n\.\!\?]+', text)
    sentence_list = [s.strip() for s in sentence_list if s.strip()]
    return " ".join(sentence_list[:sentences])


## Dynamic Users Setup
try:
    raw_records = sheet.get_all_records()
    USERS = sorted({rec['User'] for rec in raw_records if rec.get('User')})
except Exception:
    USERS = []

## User Login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.sidebar.title("🔒 選擇或新增使用者 / Select or add user")
    username = st.sidebar.selectbox("使用者 / User", USERS)
    new_user = st.sidebar.text_input("或輸入新使用者 / Or type new user")
    if new_user:
        username = new_user.strip()
    if st.sidebar.button("登入 / Login"):
        if username:
            st.session_state.logged_in = True
            st.session_state.user = username
            if username not in USERS:
                # Append a new row with empty fields for the new user
                sheet.append_row([username, datetime.date.today().strftime("%Y-%m-%d")] + [""]*8)
            components.html(""" window.location.reload(); """, height=0)
            st.stop()
        else:
            st.sidebar.error("請輸入使用者名稱 / Enter a user name")
    st.stop()
else:
    user = st.session_state.user
    st.sidebar.success(f"已登入: {user}")

## Title and Description
st.title("🌀 迷惘但想搞懂的我 / Lost but Learning")
st.markdown("黑白極簡，但情緒滿載 / Minimalist B&W, Full of Emotion")

## Input Form
today = datetime.date.today().strftime("%Y-%m-%d")
doing_today = st.text_area("📌 今天你做了什麼 / What did you do today?", height=150)
feeling_event = st.text_area("🎯 今天有感覺的事 / What felt meaningful today?")
overall_feeling = st.slider("📊 今天整體感受 (1-10)", 1, 10, 5)
self_choice = st.text_area("🧠 是自主選擇嗎？/ Was it your choice?")
dont_repeat = st.text_area("🚫 今天最不想再來的事 / What you wouldn't repeat?")
plan_tomorrow = st.text_area("🌱 明天想做什麼 / Plans for tomorrow?")
tags = st.text_area("🏷️ 標籤 / Tags (comma-separated)")
uploaded_files = st.file_uploader("📎 附加檔案 / Attachments (image or audio)", accept_multiple_files=True)

if st.button("提交 / Submit"):
    # Save attachments and get a semicolon‑separated string of file paths
    attachments_str = save_attachments(user, today, uploaded_files)
    # Determine tomorrow's tasks as a list (split by newline)
    tasks_list = [t.strip() for t in plan_tomorrow.split("\n") if t.strip()]
    # Initialize statuses: "0" means unchecked
    statuses_str = ";".join(["0"] * len(tasks_list)) if tasks_list else ""
    # Generate a summary of today's entry using the summarizer
    summary_text = generate_summary("\n".join([
        doing_today,
        feeling_event,
        self_choice,
        dont_repeat,
        plan_tomorrow
    ]))
    # Assemble the row for insertion.  Ensure the row matches the header length by
    # including task statuses and summary as the last two fields.
    row = [user, today, doing_today, feeling_event, overall_feeling, self_choice,
           dont_repeat, plan_tomorrow, tags, attachments_str, statuses_str, summary_text]
    sheet.append_row(row)
    st.balloons()
    st.success("已送出！明天見🎉")
    st.markdown("---")

## Read data once for the rest of the page
try:
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    # Guarantee presence of additional columns
    for col in ['Task Statuses', 'Summary']:
        if col not in df.columns:
            df[col] = ''
except Exception as e:
    st.error(f"讀取紀錄時發生錯誤：{e}")
    df = pd.DataFrame()

## On This Day – show past entries from the same date in previous years
st.markdown("---")
st.subheader("🕰️ 歷史上的今天 / On This Day")
if not df.empty:
    today_month_day = today[5:]  # 'MM-DD'
    past_entries = df[(df['User'] == user) & (df['Date'].str[5:] == today_month_day) & (df['Date'] != today)]
    if not past_entries.empty:
        for _, row in past_entries.iterrows():
            st.markdown(f"""
**{row['Date']}** – {render_multiline(row.get('What did you do today?', ''))}
""", unsafe_allow_html=True)
    else:
        st.info("今天沒有往年的紀錄。")
else:
    st.info("目前沒有任何紀錄。")

## History Section with tag filter and sentiment analysis
st.markdown("---")
st.subheader("📜 歷史紀錄（最近10筆）")

if not df.empty:
    df_user = df[df['User'] == user].copy()
    # Keep track of the sheet row number (index + 2 because header occupies row 1)
    df_user['_row_number'] = df_user.index + 2
    # Parse tags into sets for filtering
    df_user['TagList'] = df_user['Tags'].fillna('').apply(lambda t: [tag.strip() for tag in t.split(',') if tag.strip()])
    # Parse tasks and statuses
    def parse_tasks_and_statuses(plans: str, statuses: str):
        tasks_list = [t.strip() for t in plans.split('\n') if t.strip()]
        status_list = [s.strip() for s in statuses.split(';') if s.strip()]
        # Align status list length with tasks; default to '0' (unchecked)
        if len(status_list) < len(tasks_list):
            status_list += ['0'] * (len(tasks_list) - len(status_list))
        elif len(status_list) > len(tasks_list):
            status_list = status_list[:len(tasks_list)]
        bool_status = [s == '1' for s in status_list]
        return tasks_list, bool_status
    df_user[['TaskList', 'TaskStatusList']] = df_user.apply(
        lambda r: parse_tasks_and_statuses(r.get('Plans for tomorrow', ''), r.get('Task Statuses', '')), axis=1, result_type='expand'
    )
    # Build unique tag options
    unique_tags = sorted({tag for tags_list in df_user['TagList'] for tag in tags_list})
    selected_tags = st.multiselect("選擇標籤過濾 / Filter by tags", options=unique_tags)
    if selected_tags:
        df_user = df_user[df_user['TagList'].apply(lambda tags_list: any(tag in tags_list for tag in selected_tags))]
    # Limit to last 10 entries
    df_user = df_user.sort_values('Date').tail(10)
    # Display each entry
    for idx, row in df_user.iterrows():
        sentiment = sentiment_score(row.get('What did you do today?', ''))
        sentiment_label = {1: 'Positive', -1: 'Negative', 0: 'Neutral'}.get(sentiment, 'Neutral')
        st.markdown(f"""
🗓️ **日期：** {row.get('Date', '')}  
📌 **今天你做了什麼 / What did you do today?**： {render_multiline(row.get('What did you do today?', ''))}  
🎯 **今天有感覺的事 / What felt meaningful today?**： {render_multiline(row.get('Meaningful Event', ''))}  
📊 **今天整體感受 (1-10)**： {row.get('Mood', '')}/10  
🧠 **是自主選擇嗎？/ Was it your choice?**： {render_multiline(row.get('Was it your choice?', ''))}  
🚫 **今天最不想再來的事 / What you wouldn't repeat?**： {render_multiline(row.get('What you wouldn’t repeat', ''))}  
🌱 **明天想做什麼 / Plans for tomorrow?**： {render_multiline(row.get('Plans for tomorrow', ''))}  
🏷️ **標籤 / Tags**： {', '.join(row['TagList'])}  
💬 **Sentiment (auto)**： {sentiment_label}
""", unsafe_allow_html=True)
        # Display attachments if any
        attach_str = row.get('Attachments', '')
        if attach_str:
            files = [p for p in attach_str.split(';') if p]
            for fpath in files:
                if fpath.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')) and os.path.exists(fpath):
                    st.image(fpath, width=200)
                elif fpath.lower().endswith(('.mp3', '.wav', '.ogg')) and os.path.exists(fpath):
                    st.audio(fpath)
        # Display tomorrow's plans as checkboxes (habit/task management)
        tasks = row['TaskList'] if isinstance(row['TaskList'], list) else []
        statuses = row['TaskStatusList'] if isinstance(row['TaskStatusList'], list) else []
        if tasks:
            st.write("**Tomorrow's Tasks:**")
            for i, t in enumerate(tasks):
                st.checkbox(
                    t,
                    value=statuses[i] if i < len(statuses) else False,
                    key=f"task-{row['_row_number']}-{i}"
                )
        # Display summary if available
        if row.get('Summary'):
            st.write(f"**摘要 / Summary:** {render_multiline(row['Summary'])}")

    # Provide a button to persist task statuses back to the sheet
    if not df_user.empty and st.button("💾 保存任務狀態 / Save Task Statuses"):
        # Iterate over displayed entries and record the checkbox states
        for _, row in df_user.iterrows():
            tasks = row['TaskList'] if isinstance(row['TaskList'], list) else []
            # Build a list of "1"/"0" strings representing checked/unchecked
            statuses_list = []
            for i, _ in enumerate(tasks):
                key = f"task-{row['_row_number']}-{i}"
                checked = st.session_state.get(key, False)
                statuses_list.append('1' if checked else '0')
            statuses_str = ';'.join(statuses_list)
            # Update in-memory DataFrame
            df.loc[row.name, 'Task Statuses'] = statuses_str
            # Update the sheet cell
            try:
                row_number = row['_row_number']
                # Determine column index for Task Statuses
                if 'Task Statuses' in HEADER_ROW:
                    col_idx = HEADER_ROW.index('Task Statuses') + 1
                else:
                    col_idx = len(HEADER_ROW) + 1
                sheet.update_cell(row_number, col_idx, statuses_str)
            except Exception as e:
                st.error(f"更新任務狀態時發生錯誤：{e}")
        st.success("任務狀態已儲存！")

    # Visual analytics – weekly mood bar chart
    st.markdown("---")
    st.subheader("📊 每週心情平均 / Average Mood by Week")
    df_user['Date_dt'] = pd.to_datetime(df_user['Date'])
    df_user['Week'] = df_user['Date_dt'].dt.to_period('W').astype(str)
    weekly_mood = df_user.groupby('Week')['Mood'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean()).dropna()
    if not weekly_mood.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        weekly_mood.plot(kind='bar', ax=ax2, color='skyblue')
        ax2.set_title('Average Mood by Week')
        ax2.set_xlabel('Week')
        ax2.set_ylabel('Average Mood')
        st.pyplot(fig2)
    else:
        st.info("沒有足夠的資料繪製每週心情圖表。")
else:
    st.info("目前還沒有紀錄喔。")

## Export Section
st.markdown("---")
st.subheader("📤 匯出紀錄 / Export Entries")

export_option = st.radio("選擇要匯出的內容 / Choose what to export:", [
    "🔹 單日紀錄 / One Day (Current User)",
    "🔸 最近10筆 / Recent 10 Entries (Current User)",
    "🔺 所有紀錄 / All Entries (All Users)"
])

export_format = st.radio("選擇格式 / Choose format:", [
    "CSV", "JSON"
], index=0, horizontal=True)

if not df.empty:
    if export_option == "🔹 單日紀錄 / One Day (Current User)":
        export_date = st.selectbox("選擇日期 / Select a date", df_user['Date_dt'].dt.strftime('%Y-%m-%d').tolist())
        export_df = df_user[df_user['Date_dt'].dt.strftime('%Y-%m-%d') == export_date]
    elif export_option == "🔸 最近10筆 / Recent 10 Entries (Current User)":
        export_df = df_user.tail(10)
    else:
        export_df = pd.DataFrame(data)

    # Prepare data for export; convert lists to strings
    export_df_copy = export_df.copy()
    export_df_copy['Tags'] = export_df_copy['TagList'].apply(lambda lst: ', '.join(lst))
    export_df_copy.drop(columns=['TagList', 'Date_dt', 'Week'], errors='ignore', inplace=True)

    if export_format == "CSV":
        csv_data = export_df_copy.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下載 CSV / Download CSV",
            data=csv_data,
            file_name="journal_export.csv",
            mime='text/csv'
        )
    else:
        # JSON export
        json_str = export_df_copy.to_json(orient='records', force_ascii=False)
        json_bytes = json_str.encode('utf-8')
        st.download_button(
            label="📥 下載 JSON / Download JSON",
            data=json_bytes,
            file_name="journal_export.json",
            mime='application/json'
        )
else:
    st.info("沒有資料可匯出。")
