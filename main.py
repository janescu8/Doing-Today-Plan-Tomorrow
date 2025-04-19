import streamlit as st
import datetime
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt

# --- Google Sheets Setup ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(st.secrets["google_auth"], scopes=scope)
client = gspread.authorize(creds)
sheet = client.open("迷惘但想搞懂的我").sheet1

# --- Streamlit UI ---
st.set_page_config(page_title="🌀 迷惘但想搞懂的我", layout="centered")
# Title in bilingual
st.title("🌀 迷惘但想搞懂的我 / Lost but Learning")
st.markdown("黑白極簡，但情緒滿載 / Minimalist B&W, Full of Emotion")

# Get today's date
today = datetime.date.today().strftime("%Y-%m-%d")

# --- Input Fields ---
doing_today = st.text_input("📌 今天你做了什麼 / What did you do today?")
feeling_event = st.text_input("🎯 今天你有感覺的事 / What felt meaningful today?")
overall_feeling = st.slider("📊 今天整體感受 / Overall feeling today (1-10)", 1, 10, 5)
self_choice = st.text_input("🧠 今天做的事，是自己選的嗎？ / Was today’s choice yours?")
dont_repeat = st.text_input("🚫 今天最不想再來一次的事 / What you wouldn’t want to repeat today?")
plan_tomorrow = st.text_input("🌱 明天你想做什麼 / What do you plan for tomorrow?")

if st.button("提交 / Submit"):
    row = [today, doing_today, feeling_event, overall_feeling, self_choice, dont_repeat, plan_tomorrow]
    sheet.append_row(row)
    st.balloons()
    st.success("資料已送出，明天還記得來哦。/ Submitted! See you tomorrow.")

    # Show submitted
    st.markdown("---")
    st.subheader("🎉 你今天記錄的是 / Today's entry:")
    st.write(f"📅 {today}")
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
        # Ensure correct column names
        df.columns = [col.strip() for col in df.columns]
        # Show last 20 entries
        recent = df.tail(20)
        # Display each entry
        for _, row in recent.iterrows():
            st.markdown(f"""
            <div style='border:1px solid #666; border-radius:8px; padding:8px; margin-bottom:8px;'>
                <strong>📅 日期 / Date:</strong> {row.get('日期', '')}<br>
                <strong>📌 做了什麼 / Doing:</strong> {row.get('今天你做了什麼', '')}<br>
                <strong>🎯 感覺 / Feeling:</strong> {row.get('今天你有感覺的事', '')}<br>
                <strong>📊 感受 / Mood:</strong> {row.get('今天整體感受（1～10）', '')}/10<br>
                <strong>🧠 自選 / Self-choice:</strong> {row.get('今天做的事，是自己選的嗎？', '')}<br>
                <strong>🚫 不想再來 / Don’t repeat:</strong> {row.get('今天最不想再來一次的事：', '')}<br>
                <strong>🌱 明日計畫 / Plan:</strong> {row.get('明天你想做什麼', '')}
            </div>
            """, unsafe_allow_html=True)

        # --- Mood Log and Trend ---
        st.markdown("---")
        st.subheader("📈 Mood Log & Trend / 心情記錄與趨勢圖")
        # Parse dates and mood
        mood_df = df[['日期', '今天整體感受（1～10）']].tail(11).copy()
        mood_df.columns = ['date', 'mood']
        mood_df['date'] = pd.to_datetime(mood_df['date'])
        mood_df['mood'] = pd.to_numeric(mood_df['mood'], errors='coerce')
        mood_df = mood_df.dropna()
        mood_df = mood_df.sort_values('date')

        # Display table
        st.table(mood_df.rename(columns={'date': '日期 / Date', 'mood': '感受 / Mood'}).assign(日期=lambda x: x['日期 / Date'].dt.strftime('%Y-%m-%d')))

        # Plot trend
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
