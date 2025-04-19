import streamlit as st
import datetime
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# --- User Setup ---
USERS = ["Sanny"]  # 可自訂使用者清單

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.sidebar.title("🔒 請選擇或輸入使用者名稱 / Select or enter your username")
    username = st.sidebar.selectbox("使用者名稱 / Username", USERS)
    new_user = st.sidebar.text_input("或輸入新名稱 / Or enter a new username")
    if new_user:
        username = new_user.strip()
    if st.sidebar.button("登入 / Login"):
        if username:
            st.session_state.logged_in = True
            st.session_state.user = username
            components.html("""<script>window.location.reload();</script>""", height=0)
            st.stop()
        else:
            st.sidebar.error("請輸入有效的使用者名稱 / Please provide a valid username")
    st.stop()
else:
    user = st.session_state.user

# --- Google Sheets Setup ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(st.secrets["google_auth"], scopes=scope)
client = gspread.authorize(creds)
sheet = client.open("迷惘但想搞懂的我").sheet1

# --- Streamlit UI ---
st.set_page_config(page_title="🌀 迷惘但想搞懂的我", layout="centered")
st.title("🌀 迷惘但想搞懂的我 / Lost but Learning")
st.markdown("黑白極簡，但情緒滿載 / Minimalist B&W, Full of Emotion")
st.sidebar.success(f"已登入 / Logged in: {user}")

# --- Input ---
today = datetime.date.today().strftime("%Y-%m-%d")
doing_today = st.text_input("📌 今天你做了什麼 / What did you do today?")
feeling_event = st.text_input("🎯 今天有感覺的事 / What felt meaningful today?")
overall_feeling = st.slider("📊 今天整體感受 (1-10) / Overall feeling today", 1, 10, 5)
self_choice = st.text_input("🧠 今天做的事，是自己選的嗎？/ Was today’s choice yours?")
dont_repeat = st.text_input("🚫 今天最不想再來一次的事 / What you wouldn’t want to repeat today?")
plan_tomorrow = st.text_input("🌱 明天你想做什麼 / What do you plan for tomorrow?")

if st.button("提交 / Submit"):
    row = [user, today, doing_today, feeling_event, overall_feeling, self_choice, dont_repeat, plan_tomorrow]
    sheet.append_row(row)
    st.balloons()
    st.success("資料已送出，明天還記得來哦。/ Submitted! See you tomorrow.")
    st.markdown("---")
    st.subheader("🎉 你今天記錄的是 / Today's entry:")
    st.write(f"👤 使用者 / User: {user}")
    st.write(f"📅 日期 / Date: {today}")
    st.write(f"📌 {doing_today}")
    st.write(f"🎯 {feeling_event}")
    st.write(f"📊 {overall_feeling}/10")
    st.write(f"🧠 {self_choice}")
    st.write(f"🚫 {dont_repeat}")
    st.write(f"🌱 {plan_tomorrow}")

# --- Display History and Mood Log ---
st.markdown("---")
st.subheader("📜 歷史紀錄 (最近20筆) / History (Last 20)")
try:
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    if not df.empty:
        df.columns = ['使用者', '日期', '今天你做了什麼', '今天有感覺的事', '今天整體感受',
                      '今天做的事，是自己選的嗎？', '今天最不想再來一次的事', '明天你想做什麼']
        if user != 'admin':
            df = df[df['使用者'] == user]
        recent = df.tail(20)
        for _, row in recent.iterrows():
            st.markdown(f"""
            <div style='border:1px solid #666; border-radius:8px; padding:8px; margin-bottom:8px;'>
                <strong>👤 使用者 / User:</strong> {row['使用者']}<br>
                <strong>📅 日期 / Date:</strong> {row['日期']}<br>
                <strong>📌 做了什麼 / Doing:</strong> {row['今天你做了什麼']}<br>
                <strong>🎯 感覺 / Feeling:</strong> {row['今天有感覺的事']}<br>
                <strong>📊 感受 / Mood:</strong> {row['今天整體感受']}/10<br>
                <strong>🧠 自選 / Self-choice:</strong> {row['今天做的事，是自己選的嗎？']}<br>
                <strong>🚫 不想再來 / Don’t repeat:</strong> {row['今天最不想再來一次的事']}<br>
                <strong>🌱 明日計畫 / Plan:</strong> {row['明天你想做什麼']}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📈 Mood Log & Trend / 心情記錄與趨勢圖")
        mood_df = df[['日期', '今天整體感受']].tail(11).copy()
        mood_df.columns = ['date', 'mood']
        mood_df['date'] = pd.to_datetime(mood_df['date'])
        mood_df['mood'] = pd.to_numeric(mood_df['mood'], errors='coerce')
        mood_df = mood_df.dropna().sort_values('date')

        st.table(mood_df.assign(date=lambda x: x['date'].dt.strftime('%Y-%m-%d')).rename(columns={'date':'日期 / Date','mood':'感受 / Mood'}))

        fig, ax = plt.subplots()
        ax.plot(mood_df['date'], mood_df['mood'], marker='o')
        ax.set_title('Mood Trend Over Time / 心情趨勢')
        ax.set_xlabel('Date / 日期')
        ax.set_ylabel('Mood (1-10) / 感受')
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
        fig.autofmt_xdate()
        st.pyplot(fig)
    else:
        st.info("目前還沒有紀錄喔 / No entries yet.")
except Exception as e:
    st.error(f"讀取紀錄時發生錯誤：{e}")
