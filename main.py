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
    df = pd.DataFrame(raw_records)
    # Normalize column names
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
    if col_map:
        df.rename(columns=col_map, inplace=True)

    if df.empty:
        st.info("目前還沒有紀錄喔 / No entries yet.")
    else:
        if 'admin' not in st.session_state or st.session_state.user != 'admin':
            user_filter = st.session_state.get('user', None)
            if user_filter:
                df = df[df['使用者'] == user_filter]
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

        # 顯示表格
        st.table(
            mood_df.assign(date=lambda x: x['date'].dt.strftime('%Y-%m-%d'))
                   .rename(columns={'date':'日期 / Date','mood':'感受 / Mood'})
        )

        # 使用純 pyplot 繪製圖表
        plt.figure(figsize=(10, 4))
        plt.plot(mood_df['date'], mood_df['mood'], marker='o')
        plt.title('Mood Trend Over Time / 心情趨勢')
        plt.xlabel('Date / 日期')
        plt.ylabel('Mood (1-10) / 感受')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gcf().autofmt_xdate()
        st.pyplot(plt)

except Exception as e:
    st.error(f"讀取紀錄時發生錯誤：{e}")
