
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
sheet = client.open("journal_export").sheet1

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
            if username not in USERS:
                sheet.append_row([username, datetime.date.today().strftime("%Y-%m-%d")] + [""]*7)
            components.html("""<script>window.location.reload();</script>""", height=0)
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

# --- 顯示過去紀錄與趨勢圖 ---
# 換行處理函數
def render_multiline(text):
    return text.replace('\n', '<br>')

st.markdown("---")
st.subheader("📜 歷史紀錄（最近10筆）")

try:
    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    if not df.empty:
        df = df[df['使用者'] == user].tail(10)
        for index, row in df.iterrows():
            st.markdown(f"""
            <div style='border:1px solid #ccc; border-radius:10px; padding:10px; margin-bottom:10px;'>
                <strong>🗓️ 日期：</strong> {row.get('日期', '')}<br>
                <strong>📌 今天你做了什麼 / What did you do today?：</strong><br> {render_multiline(row.get('今天你做了什麼', ''))}<br>
                <strong>🎯 今天有感覺的事 / What felt meaningful today?：</strong><br> {render_multiline(row.get('今天你有感覺的事', ''))}<br>
                <strong>📊 今天整體感受 (1-10)：</strong> {row.get('今天整體感受', '')}/10<br>
                <strong>🧠 是自主選擇嗎？/ Was it your choice?：</strong><br> {render_multiline(row.get('今天做的事，是自己選的嗎？', ''))}<br>
                <strong>🚫 今天最不想再來的事 / What you wouldn't repeat?：</strong><br> {render_multiline(row.get('今天最不想再來一次的事', ''))}<br>
                <strong>🌱 明天想做什麼 / Plans for tomorrow?：</strong><br> {render_multiline(row.get('明天你想做什麼', ''))}<br>
                <strong>🏷️ 標籤 / Tags：</strong><br> {render_multiline(row.get('標籤', ''))}<br>
            </div>
            """, unsafe_allow_html=True)

        # 心情趨勢圖
        st.markdown("---")
        st.subheader("📈 心情趨勢圖 / Mood Trend")
        mood_df = df[['日期', '今天整體感受']].copy()
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

# --- 編輯紀錄區塊 ---
st.markdown("---")
st.subheader("✏️ 編輯紀錄 / Edit Past Entry")

# 取出該使用者所有紀錄
user_data = df[df['使用者'] == user].copy()
user_data['日期'] = pd.to_datetime(user_data['日期'])
user_data = user_data.sort_values('日期', ascending=False).reset_index(drop=True)

if not user_data.empty:
    edit_dates = user_data['日期'].dt.strftime('%Y-%m-%d').tolist()
    default_date = edit_dates[0]  # 預設是最新一筆
    selected_date = st.selectbox("選擇要編輯的日期 / Select a date to edit", edit_dates, index=0)

    # 找到要編輯的那筆紀錄
    record_to_edit = user_data[user_data['日期'].dt.strftime('%Y-%m-%d') == selected_date].iloc[0]
    row_number_in_sheet = df[(df['使用者'] == user) & (df['日期'] == selected_date)].index[0] + 2  # gspread row (加上 header)

    # 建立可編輯表單
    with st.form("edit_form"):
        new_doing = st.text_area("📌 今天你做了什麼 / What did you do today?", record_to_edit.get('今天你做了什麼', ''), height=100)
        new_event = st.text_area("🎯 今天有感覺的事 / What felt meaningful today?", record_to_edit.get('今天你有感覺的事', ''))
        new_mood = st.slider("📊 今天整體感受 (1-10)", 1, 10, int(record_to_edit.get('今天整體感受', 5)))
        new_choice = st.text_area("🧠 是自主選擇嗎？/ Was it your choice?", record_to_edit.get('今天做的事，是自己選的嗎？', ''))
        new_repeat = st.text_area("🚫 今天最不想再來的事 / What you wouldn't repeat?", record_to_edit.get('今天最不想再來一次的事', ''))
        new_plan = st.text_area("🌱 明天想做什麼 / Plans for tomorrow?", record_to_edit.get('明天你想做什麼', ''))
        new_tags = st.text_area("🏷️ 標籤 / Tags (comma-separated)", record_to_edit.get('標籤', ''))

        submitted = st.form_submit_button("更新紀錄 / Update Entry")
        if submitted:
            updated_row = [user, selected_date, new_doing, new_event, new_mood, new_choice, new_repeat, new_plan, new_tags]
            sheet.update(f'A{row_number_in_sheet}:I{row_number_in_sheet}', [updated_row])
            st.success(f"{selected_date} 的紀錄已成功更新！ / Entry Updated")
            st.rerun()
else:
    st.info("目前尚無可供編輯的紀錄。")
    
# --- 搜尋紀錄功能 / Search Entries ---
st.markdown("---")
st.subheader("🔍 搜尋紀錄 / Search Journal Entries")

search_query = st.text_input("輸入關鍵字來搜尋所有紀錄 / Enter keyword to search all entries")

if search_query:
    try:
        all_data = sheet.get_all_records()
        search_df = pd.DataFrame(all_data)
        search_df = search_df.fillna("")  # Handle NaN for searching

        # 將所有欄位轉為字串並搜尋關鍵字
        mask = search_df.apply(lambda row: row.astype(str).str.contains(search_query, case=False, na=False)).any(axis=1)
        result_df = search_df[mask]

        if not result_df.empty:
            st.success(f"找到 {len(result_df)} 筆包含「{search_query}」的紀錄")
            for index, row in result_df.iterrows():
                st.markdown(f"""
                <div style='border:1px solid #ccc; border-radius:10px; padding:10px; margin-bottom:10px;'>
                    <strong>🗓️ 日期：</strong> {row.get('日期', '')}<br>
                    <strong>📌 今天你做了什麼 / What did you do today?：</strong><br> {render_multiline(row.get('今天你做了什麼', ''))}<br>
                    <strong>🎯 今天有感覺的事 / What felt meaningful today?：</strong><br> {render_multiline(row.get('今天你有感覺的事', ''))}<br>
                    <strong>📊 今天整體感受 (1-10)：</strong> {row.get('今天整體感受', '')}/10<br>
                    <strong>🧠 是自主選擇嗎？/ Was it your choice?：</strong><br> {render_multiline(row.get('今天做的事，是自己選的嗎？', ''))}<br>
                    <strong>🚫 今天最不想再來的事 / What you wouldn't repeat?：</strong><br> {render_multiline(row.get('今天最不想再來一次的事', ''))}<br>
                    <strong>🌱 明天想做什麼 / Plans for tomorrow?：</strong><br> {render_multiline(row.get('明天你想做什麼', ''))}<br>
                    <strong>🏷️ 標籤 / Tags：</strong> {row.get('標籤', '')}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"沒有找到包含「{search_query}」的紀錄。")

    except Exception as e:
        st.error(f"搜尋時發生錯誤：{e}")


# --- 匯出資料為 CSV ---
st.markdown("---")
st.subheader("📤 匯出紀錄 / Export Entries as CSV")

export_option = st.radio("選擇要匯出的內容 / Choose what to export:", [
    "🔹 單日紀錄 / One Day (Current User)",
    "🔸 最近10筆 / Recent 10 Entries (Current User)",
    "🔺 所有紀錄 / All Entries (All Users)"
])

if export_option == "🔹 單日紀錄 / One Day (Current User)":
    export_date = st.selectbox("選擇日期 / Select a date", user_data['日期'].dt.strftime('%Y-%m-%d').tolist())
    export_df = user_data[user_data['日期'].dt.strftime('%Y-%m-%d') == export_date]

elif export_option == "🔸 最近10筆 / Recent 10 Entries (Current User)":
    export_df = user_data.head(10)

elif export_option == "🔺 所有紀錄 / All Entries (All Users)":
    all_data = sheet.get_all_records()
    export_df = pd.DataFrame(all_data)

    # Rename columns only if they exist in the full sheet
    export_df.rename(columns={
        '使用者': 'User',
        '日期': 'Date',
        '今天你做了什麼': 'What did you do today?',
        '今天有感覺的事': 'Meaningful Event',
        '今天整體感受': 'Mood',
        '今天做的事，是自己選的嗎？': 'Was it your choice?',
        '今天最不想再來一次的事': 'What you wouldn’t repeat',
        '明天你想做什麼': 'Plans for tomorrow',
        '標籤': 'Tags'  # 🏷️ 新增這一行
    }, inplace=True)


# Export CSV
csv = export_df.to_csv(index=False).encode('utf-8-sig')  # UTF-8 with BOM
st.download_button(
    label="📥 下載 CSV / Download CSV",
    data=csv,
    file_name="journal_export.csv",
    mime='text/csv'
)

