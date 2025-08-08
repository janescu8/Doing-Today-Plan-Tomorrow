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


# --- Google Sheets Setup ---
# This version of the app expects the worksheet to use **English column names**
# (User, Date, What did you do today?, Meaningful Event, Mood, Was it your choice?,
# What you wouldn’t repeat, Plans for tomorrow, Tags).  If your worksheet still
# uses the older Chinese headers, rename them in Google Sheets before running
# this code.
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(st.secrets["google_auth"], scopes=scope)
client = gspread.authorize(creds)
sheet = client.open("journal_export").sheet1

# --- Dynamic Users Setup ---
try:
    raw_records = sheet.get_all_records()
    # Collect unique user names from the 'User' column
    USERS = sorted({rec['User'] for rec in raw_records if rec.get('User')})
except Exception:
    USERS = []

# --- User Login ---
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
            # Append a blank row with English headers if the user is new
            if username not in USERS:
                sheet.append_row([username, datetime.date.today().strftime("%Y-%m-%d")] + [""]*7)
            components.html(""" window.location.reload(); """, height=0)
            st.stop()
        else:
            st.sidebar.error("請輸入使用者名稱 / Enter a user name")
    st.stop()
else:
    user = st.session_state.user
    st.sidebar.success(f"已登入: {user}")

# --- Title and Description ---
st.title("🌀 迷惘但想搞懂的我 / Lost but Learning")
st.markdown("黑白極簡，但情緒滿載 / Minimalist B&W, Full of Emotion")

# --- Input Form ---
today = datetime.date.today().strftime("%Y-%m-%d")
doing_today = st.text_area("📌 今天你做了什麼 / What did you do today?", height=150)
feeling_event = st.text_area("🎯 今天有感覺的事 / What felt meaningful today?")
overall_feeling = st.slider("📊 今天整體感受 (1-10)", 1, 10, 5)
self_choice = st.text_area("🧠 是自主選擇嗎？/ Was it your choice?")
dont_repeat = st.text_area("🚫 今天最不想再來的事 / What you wouldn't repeat?")
plan_tomorrow = st.text_area("🌱 明天想做什麼 / Plans for tomorrow?")
tags = st.text_area("🏷️ 標籤 / Tags (comma-separated)")

if st.button("提交 / Submit"):
    row = [user, today, doing_today, feeling_event, overall_feeling, self_choice, dont_repeat, plan_tomorrow, tags]
    sheet.append_row(row)
    st.balloons()
    st.success("已送出！明天見🎉")
    st.markdown("---")

# --- Display past records and mood trend ---
def render_multiline(text: str) -> str:
    """
    Replace newline characters with HTML `<br>` tags so that
    multi‑line journal entries retain their original formatting
    when displayed with `st.markdown` (with `unsafe_allow_html=True`).
    """
    # If text is not a string (e.g. NaN), convert it to empty string
    if not isinstance(text, str):
        text = str(text)
    return text.replace('\n', '<br>')

st.markdown("---")
st.subheader("📜 歷史紀錄（最近10筆）")

try:
    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    if not df.empty:
        # Filter by current user and take the last 10 entries
        df_user = df[df['User'] == user].tail(10)
        for index, row in df_user.iterrows():
            st.markdown(f"""

🗓️ 日期： {row.get('Date', '')}
📌 今天你做了什麼 / What did you do today?： {render_multiline(row.get('What did you do today?', ''))}
🎯 今天有感覺的事 / What felt meaningful today?： {render_multiline(row.get('Meaningful Event', ''))}
📊 今天整體感受 (1-10)： {row.get('Mood', '')}/10
🧠 是自主選擇嗎？/ Was it your choice?： {render_multiline(row.get('Was it your choice?', ''))}
🚫 今天最不想再來的事 / What you wouldn't repeat?： {render_multiline(row.get('What you wouldn’t repeat', ''))}
🌱 明天想做什麼 / Plans for tomorrow?： {render_multiline(row.get('Plans for tomorrow', ''))}
🏷️ 標籤 / Tags： {render_multiline(row.get('Tags', ''))}

""", unsafe_allow_html=True)

        # Mood trend chart
        st.markdown("---")
        st.subheader("📈 心情趨勢圖 / Mood Trend")
        mood_df = df_user[['Date', 'Mood']].copy()
        mood_df.columns = ['date', 'mood']
        mood_df['date'] = pd.to_datetime(mood_df['date'])
        mood_df['mood'] = pd.to_numeric(mood_df['mood'], errors='coerce')
        mood_df = mood_df.dropna().sort_values('date')

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(mood_df['date'], mood_df['mood'], marker='o')
        ax.set_title('Mood Trend Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Mood')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        fig.autofmt_xdate()
        st.pyplot(fig)
    else:
        st.info("目前還沒有紀錄喔。")

except Exception as e:
    st.error(f"讀取紀錄時發生錯誤：{e}")

# --- Edit Entry Section ---
st.markdown("---")
st.subheader("✏️ 編輯紀錄 / Edit Past Entry")

# Extract all records for the current user
user_data = df[df['User'] == user].copy()
user_data['Date'] = pd.to_datetime(user_data['Date'])
user_data = user_data.sort_values('Date', ascending=False).reset_index(drop=True)

if not user_data.empty:
    edit_dates = user_data['Date'].dt.strftime('%Y-%m-%d').tolist()
    default_date = edit_dates[0]  # default to the most recent entry
    selected_date = st.selectbox("選擇要編輯的日期 / Select a date to edit", edit_dates, index=0)

    # Locate the entry to edit
    record_to_edit = user_data[user_data['Date'].dt.strftime('%Y-%m-%d') == selected_date].iloc[0]
    row_number_in_sheet = df[(df['User'] == user) & (df['Date'] == selected_date)].index[0] + 2  # +2 for header row

    # Build editable form
    with st.form("edit_form"):
        new_doing = st.text_area("📌 今天你做了什麼 / What did you do today?", record_to_edit.get('What did you do today?', ''), height=100)
        new_event = st.text_area("🎯 今天有感覺的事 / What felt meaningful today?", record_to_edit.get('Meaningful Event', ''))
        new_mood = st.slider("📊 今天整體感受 (1-10)", 1, 10, int(record_to_edit.get('Mood', 5)))
        new_choice = st.text_area("🧠 是自主選擇嗎？/ Was it your choice?", record_to_edit.get('Was it your choice?', ''))
        new_repeat = st.text_area("🚫 今天最不想再來的事 / What you wouldn't repeat?", record_to_edit.get('What you wouldn’t repeat', ''))
        new_plan = st.text_area("🌱 明天想做什麼 / Plans for tomorrow?", record_to_edit.get('Plans for tomorrow', ''))
        new_tags = st.text_area("🏷️ 標籤 / Tags (comma-separated)", record_to_edit.get('Tags', ''))

        submitted = st.form_submit_button("更新紀錄 / Update Entry")
        if submitted:
            updated_row = [user, selected_date, new_doing, new_event, new_mood, new_choice, new_repeat, new_plan, new_tags]
            sheet.update(f'A{row_number_in_sheet}:I{row_number_in_sheet}', [updated_row])
            st.success(f"{selected_date} 的紀錄已成功更新！ / Entry Updated")
            st.rerun()
else:
    st.info("目前尚無可供編輯的紀錄。")

# --- Search Entries Function ---
st.markdown("---")
st.subheader("🔍 搜尋紀錄 / Search Journal Entries")

search_query = st.text_input("輸入關鍵字來搜尋所有紀錄 / Enter keyword to search all entries")

if search_query:
    try:
        all_data = sheet.get_all_records()
        search_df = pd.DataFrame(all_data)
        search_df = search_df.fillna("")  # handle NaN for searching

        # Search across all columns by converting to string
        mask = search_df.apply(lambda row: row.astype(str).str.contains(search_query, case=False, na=False)).any(axis=1)
        result_df = search_df[mask]

        if not result_df.empty:
            st.success(f"找到 {len(result_df)} 筆包含「{search_query}」的紀錄")
            for index, row in result_df.iterrows():
                st.markdown(f"""

🗓️ 日期： {row.get('Date', '')}
📌 今天你做了什麼 / What did you do today?： {render_multiline(row.get('What did you do today?', ''))}
🎯 今天有感覺的事 / What felt meaningful today?： {render_multiline(row.get('Meaningful Event', ''))}
📊 今天整體感受 (1-10)： {row.get('Mood', '')}/10
🧠 是自主選擇嗎？/ Was it your choice?： {render_multiline(row.get('Was it your choice?', ''))}
🚫 今天最不想再來的事 / What you wouldn't repeat?： {render_multiline(row.get('What you wouldn’t repeat', ''))}
🌱 明天想做什麼 / Plans for tomorrow?： {render_multiline(row.get('Plans for tomorrow', ''))}
🏷️ 標籤 / Tags： {row.get('Tags', '')}

""", unsafe_allow_html=True)
        else:
            st.info(f"沒有找到包含「{search_query}」的紀錄。")

    except Exception as e:
        st.error(f"搜尋時發生錯誤：{e}")

# --- Export Data as CSV ---
st.markdown("---")
st.subheader("📤 匯出紀錄 / Export Entries as CSV")

export_option = st.radio("選擇要匯出的內容 / Choose what to export:", [
    "🔹 單日紀錄 / One Day (Current User)",
    "🔸 最近10筆 / Recent 10 Entries (Current User)",
    "🔺 所有紀錄 / All Entries (All Users)"
])

if export_option == "🔹 單日紀錄 / One Day (Current User)":
    export_date = st.selectbox("選擇日期 / Select a date", user_data['Date'].dt.strftime('%Y-%m-%d').tolist())
    export_df = user_data[user_data['Date'].dt.strftime('%Y-%m-%d') == export_date]

elif export_option == "🔸 最近10筆 / Recent 10 Entries (Current User)":
    export_df = user_data.head(10)

elif export_option == "🔺 所有紀錄 / All Entries (All Users)":
    all_data = sheet.get_all_records()
    export_df = pd.DataFrame(all_data)

    # no need to rename columns; they are already English

# Export CSV
csv = export_df.to_csv(index=False).encode('utf-8-sig')  # UTF-8 with BOM
st.download_button(
    label="📥 下載 CSV / Download CSV",
    data=csv,
    file_name="journal_export.csv",
    mime='text/csv'
)

# Word Cloud Section
st.markdown("---")
st.subheader("☁️ 常見詞雲 / Frequent Words Cloud")

# Build text content from selected columns
word_fields = [
    'What did you do today?',
    'Meaningful Event',
    'Was it your choice?',
    'What you wouldn’t repeat',
    'Plans for tomorrow',
    'Tags'
]
all_data = sheet.get_all_records()
all_df = pd.DataFrame(all_data)
text_data = all_df[word_fields].fillna('').apply(lambda row: ' '.join(str(val) for val in row), axis=1).str.cat(sep=' ')

# Font path (handles both local and Streamlit Cloud)
font_path = os.path.join("assets", "NotoSansCJKtc-Regular.otf")

# Check if font exists
if os.path.exists(font_path):
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            font_path=font_path,
            collocations=False
        ).generate(text_data)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    except Exception:
        st.info("詞雲生成失敗，可能是字體檔缺失或其他錯誤。")
else:
    st.info("找不到字體檔案，無法生成詞雲。")
