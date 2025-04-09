import streamlit as st
import datetime
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

# --- Google Sheets Setup ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(st.secrets["google_auth"], scopes=scope)
client = gspread.authorize(creds)
sheet = client.open("迷惘但想搞懂的我").sheet1

# --- Streamlit UI ---
st.set_page_config(page_title="迷惘但想搞懂的我", layout="centered")
st.title("🌀 迷惘但想搞懂的我")
st.markdown("黑白極簡，但情緒滿載")

today = datetime.date.today().strftime("%Y-%m-%d")

doing_today = st.text_input("今天你做了什麼")
feeling_event = st.text_input("今天你有感覺的事")
overall_feeling = st.slider("今天整體感受（1～10）", 1, 10, 5)
self_choice = st.text_input("今天做的事，是自己選的嗎？")
dont_repeat = st.text_input("今天最不想再來一次的事：")
plan_tomorrow = st.text_input("明天你想做什麼")

if st.button("提交"):
    row = [today, doing_today, feeling_event, overall_feeling, self_choice, dont_repeat, plan_tomorrow]
    sheet.append_row(row)
    st.balloons()
    st.success("資料已送出，明天還記得來哦。")

    # 顯示送出內容
    st.markdown("---")
    st.subheader("你今天記錄的是：")
    st.write(f"🗓️ 日期：{today}")
    st.write(f"📌 今天你做了什麼：{doing_today}")
    st.write(f"🎯 今天你有感覺的事：{feeling_event}")
    st.write(f"📊 今天整體感受：{overall_feeling}/10")
    st.write(f"🧠 今天做的事，是自己選的嗎？{self_choice}")
    st.write(f"🚫 今天最不想再來一次的事：{dont_repeat}")
    st.write(f"🌱 明天你想做什麼：{plan_tomorrow}")

# --- 顯示過去紀錄 ---
st.markdown("---")
st.subheader("📜 歷史紀錄（最近10筆）")
try:
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.tail(10)
        for index, row in df.iterrows():
            st.markdown(f"""
            <div style='border:1px solid #ccc; border-radius:10px; padding:10px; margin-bottom:10px;'>
                <strong>🗓️ 日期：</strong> {row.get('日期', '')}<br>
                <strong>📌 今天做了什麼：</strong> {row.get('今天你做了什麼', '')}<br>
                <strong>🎯 有感覺的事：</strong> {row.get('今天你有感覺的事', '')}<br>
                <strong>📊 整體感受：</strong> {row.get('今天整體感受（1～10）', '')}/10<br>
                <strong>🧠 是自己選的嗎：</strong> {row.get('今天做的事，是自己選的嗎？', '')}<br>
                <strong>🚫 最不想再來一次：</strong> {row.get('今天最不想再來一次的事：', '')}<br>
                <strong>🌱 明天想做什麼：</strong> {row.get('明天你想做什麼', '')}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("目前還沒有紀錄喔。")
except Exception as e:
    st.error(f"讀取紀錄時發生錯誤：{e}")

# --- Minimalist UI Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #fff;
        color: #111;
        font-family: 'Courier New', monospace;
    }
    .css-1d391kg p {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)
