import streamlit as st
import datetime
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit.components.v1 as components

# --- Google Sheets Setup ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(st.secrets["google_auth"], scopes=scope)
client = gspread.authorize(creds)
sheet = client.open("迷惘但想搞懂的我").sheet1

# --- Dynamic Users Setup ---
try:
    raw_records = sheet.get_all_records()
    USERS = sorted({rec['使用者'] for rec in raw_records if rec.get('使用者')})
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
            # 若新使用者，先 append 於 sheet 以保留欄位結構
            if username not in USERS:
                sheet.append_row([username, datetime.date.today().strftime("%Y-%m-%d")] + [""]*6)
            components.html("""<script>window.location.reload();</script>""", height=0)
            st.stop()
        else:
            st.sidebar.error("請輸入使用者名稱 / Enter a user name")
    st.stop()
else:
    user = st.session_state.user
    st.sidebar.success(f"已登入: {user}")

# --- Page Title ---
st.set_page_config(page_title="🌀 迷惘但想搞懂的我", layout="centered")
st.title("🌀 迷惘但想搞懂的我 / Lost but Learning")
st.markdown("黑白極簡，但情緒滿載 / Minimalist B&W, Full of Emotion")

# --- Input Form ---
today = datetime.date.today().strftime("%Y-%m-%d")
doing_today = st.text_input("📌 今天你做了什麼 / What did you do today?")
feeling_event = st.text_input("🎯 今天有感覺的事 / What felt meaningful today?")
overall_feeling = st.slider("📊 今天整體感受 (1-10)", 1, 10, 5)
self_choice = st.text_input("🧠 是自主選擇嗎？/ Was it your choice?")
dont_repeat = st.text_input("🚫 今天最不想再來的事 / What you wouldn't repeat?")
plan_tomorrow = st.text_input("🌱 明天想做什麼 / Plans for tomorrow?")

if st.button("提交 / Submit"):
    row = [user, today, doing_today, feeling_event, overall_feeling, self_choice, dont_repeat, plan_tomorrow]
    sheet.append_row(row)
    st.balloons()
    st.success("已送出！明天見🎉")
    st.markdown("---")

# --- History & Mood Trend ---
st.subheader("📜 歷史紀錄 (最近20筆)")
try:
    df = pd.DataFrame(sheet.get_all_records())
    # normalize columns
    col_map = {}
    for col in df.columns:
        if '使用者' in col:
            col_map[col] = '使用者'
        elif '日期' in col:
            col_map[col] = '日期'
        elif '做了什麼' in col:
            col_map[col] = '今天你做了什麼'
        elif '整體感受' in col:
            col_map[col] = '今天整體感受'
        elif '感覺' in col:
            col_map[col] = '今天有感覺的事'
        elif '自己選' in col:
            col_map[col] = '今天做的事，是自己選的嗎？'
        elif '不想再' in col:
            col_map[col] = '今天最不想再來一次的事'
        elif '明天' in col:
            col_map[col] = '明天你想做什麼'
    df.rename(columns=col_map, inplace=True)

    if df.empty:
        st.info("尚無紀錄")
    else:
        # filter user
        if user != 'admin':
            df = df[df['使用者'] == user]
        recent = df.tail(20)
        for _, row in recent.iterrows():
            st.markdown(f"""
            **{row['日期']}** — {row['今天你做了什麼']} (感受: {row['今天整體感受']}/10)<br>
            🎯 {row['今天有感覺的事']} | 🚫 {row['今天最不想再來一次的事']}<br>
            🌱 {row['明天你想做什麼']}
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📈 心情趨勢圖 / Mood Trend")
        mood_df = recent[['日期', '今天整體感受']].tail(11).copy()
        mood_df.columns = ['date', 'mood']
        mood_df['date'] = pd.to_datetime(mood_df['date'])
        mood_df['mood'] = pd.to_numeric(mood_df['mood'], errors='coerce')
        mood_df = mood_df.dropna().sort_values('date')

        # plot via pyplot
        plt.figure(figsize=(8,4))
        plt.plot(mood_df['date'], mood_df['mood'], marker='o')
        plt.title('Mood Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Mood')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gcf().autofmt_xdate()
        st.pyplot()

except Exception as e:
    st.error(f"讀取紀錄時發生錯誤：{e}")
