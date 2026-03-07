import streamlit as st
import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
import os
import time
import firebase_admin
from firebase_admin import credentials, db

st.set_page_config(
    page_title="UFP SENSE Dashboard",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# [ส่วนที่ 0: ตั้งค่าการเชื่อมต่อ Firebase]
# ==========================================
FIREBASE_CREDENTIALS_FILE = "serviceAccountKey.json" 
FIREBASE_DATABASE_URL = "https://lab10-f138a-default-rtdb.asia-southeast1.firebasedatabase.app/" 
FIREBASE_NODE_NAME = "raw_sensor_data/history"

if not firebase_admin._apps:
    try:
        if "firebase" in st.secrets:
            cred_dict = dict(st.secrets["firebase"])
            cred = credentials.Certificate(cred_dict)
        elif os.path.exists(FIREBASE_CREDENTIALS_FILE):
            cred = credentials.Certificate(FIREBASE_CREDENTIALS_FILE)
        else:
            st.sidebar.error("❌ ไม่พบกุญแจเชื่อมต่อ Firebase")
            cred = None
            
        if cred:
            firebase_admin.initialize_app(cred, {
                'databaseURL': FIREBASE_DATABASE_URL
            })
    except Exception as e:
        st.sidebar.error(f"❌ เชื่อมต่อ Firebase ไม่สำเร็จ: {e}")

def fetch_latest_firebase_data(limit=12):
    try:
        ref = db.reference(FIREBASE_NODE_NAME)
        data = ref.order_by_key().limit_to_last(limit).get()
        if data:
            records = [val for key, val in data.items()]
            return pd.DataFrame(records)
    except Exception as e:
        st.error(f"❌ ดึงข้อมูลผิดพลาด: {e}")
    return pd.DataFrame()

# ==========================================
# [ส่วนที่ 1: คลาสและโหลดโมเดล GRU]
# ==========================================
class SingleStepGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 8, num_layers: int = 1):
        super(SingleStepGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out

@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_folder = "models"
    model_data = None
    if not os.path.exists(base_folder): return model_data, device

    try:
        name_gru = "gru_latest" 
        path_gru = os.path.join(base_folder, f"{name_gru}.pth")
        if os.path.exists(path_gru):
            with open(os.path.join(base_folder, f"{name_gru}_preprocessor.pkl"), 'rb') as f: prep_gru = pickle.load(f)
            with open(os.path.join(base_folder, f"{name_gru}_scaler.pkl"), 'rb') as f: y_scale_gru = pickle.load(f)
            model_gru = SingleStepGRU(input_size=6, hidden_size=8, num_layers=1)
            state_dict = torch.load(path_gru, map_location=device)
            if 'linear.weight' in state_dict: state_dict['fc.weight'] = state_dict.pop('linear.weight')
            if 'linear.bias' in state_dict: state_dict['fc.bias'] = state_dict.pop('linear.bias')
            model_gru.load_state_dict(state_dict, strict=False)
            model_gru.to(device).eval()
            model_data = (model_gru, prep_gru, y_scale_gru, 12) 
    except Exception as e: pass
    return model_data, device

def generate_ai_insight(pm01):
    if pm01 >= 20000: return "🔴 Hazardous: Conditions detected. Please remain indoors and use air purifiers."
    elif pm01 >= 10000: return "🟠 Poor: Air quality is reduced. Turn on air purifiers."
    elif pm01 >= 1000: return "🟡 Moderate: Acceptable, but sensitive individuals should monitor."
    else: return "🟢 Excellent: Safe environment. Great time for normal activities."

# ==========================================
# [ส่วนที่ 2: หน้าตาเว็บ Streamlit UI]
# ==========================================
# ซ่อนเมนูที่ไม่จำเป็นของ Streamlit
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;} footer {visibility: hidden;}
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🌬️ UFP SENSE Dashboard")
st.markdown("Live Prediction by **GRU Model**")

gru_model_data, device = load_models()
app_mode = st.sidebar.radio("เลือกโหมดการทำงาน:", ("📡 โหมด Live (Firebase)", "📂 โหมด Test (Upload CSV)"))

st.sidebar.markdown("---")

if app_mode == "📡 โหมด Live (Firebase)":
    st.sidebar.markdown("**ดึงข้อมูลสดจากเซ็นเซอร์**")
    auto_refresh = st.sidebar.checkbox("🟢 เปิดระบบ Real-time (ดึงข้อมูลอัตโนมัติ)", value=False)

    # 💡 พระเอกอยู่ตรงนี้: st.empty() จะจองพื้นที่ไว้ แล้วอัปเดตแค่ข้อมูลข้างใน ไม่โหลดทั้งหน้า!
    placeholder = st.empty()

    if auto_refresh:
        if gru_model_data is not None:
            mod_i, prep_i, y_scaler_i, seq_len_i = gru_model_data
            
            while True:
                input_df = fetch_latest_firebase_data(limit=seq_len_i)
                if not input_df.empty and len(input_df) >= seq_len_i:
                    rename_map = {'wind_dir': 'Wind_Dir', 'wind_speed': 'Wind_Speed', 'outdoor_temp': 'Outdoor_Temperature', 
                                  'outdoor_hum': 'Outdoor_Humidity', 'bar': 'Bar', 'outdoor_pm25': 'Outdoor_PM2.5'}
                    input_df = input_df.rename(columns=rename_map)
                    model_cols = ['Wind_Dir', 'Wind_Speed', 'Outdoor_Temperature', 'Outdoor_Humidity', 'Bar', 'Outdoor_PM2.5']
                    
                    if all(col in input_df.columns for col in model_cols):
                        X_proc_all = prep_i.transform(input_df[model_cols])
                        seq_tensor = torch.FloatTensor(X_proc_all).unsqueeze(0).to(device) 
                        mod_i.eval()
                        with torch.no_grad(): out = mod_i(seq_tensor)
                        
                        raw_pred = y_scaler_i.inverse_transform(out.cpu().numpy())
                        pred_val = int(raw_pred[0][0] / 1000) 
                        
                        last_row = input_df.iloc[-1]
                        pm25_disp = round(last_row['Outdoor_PM2.5'], 2)
                        temp_disp = round(last_row['Outdoor_Temperature'], 2)
                        humid_disp = round(last_row['Outdoor_Humidity'], 2)
                        wind_spd = round(last_row['Wind_Speed'], 2)
                        
                        # 💡 เปลี่ยนหน้าตา UI ตรงนี้ โดยไม่ต้องโหลดหน้าเว็บใหม่
                        with placeholder.container():
                            st.info(f"**AI Insight:** {generate_ai_insight(pred_val)}")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("🏠 Indoor PM0.1 (Predicted)", f"{pred_val} PC/cm³", "Live Update", delta_color="normal")
                            col2.metric("🌫️ Outdoor PM2.5", f"{pm25_disp} µg/m³")
                            col3.metric("🌡️ Temperature", f"{temp_disp} °C")
                            col4.metric("💧 Humidity", f"{humid_disp} %")
                            
                            st.markdown("---")
                            st.subheader("📊 แนวโน้มข้อมูลล่าสุด")
                            # จำลองกราฟสั้นๆ 12 แถวล่าสุดมาโชว์ (สมูทมาก ไม่กระพริบ)
                            st.line_chart(input_df[['Outdoor_PM2.5', 'Outdoor_Temperature']])

                time.sleep(2) # รอ 2 วิ แล้ววนลูปดึงข้อมูลใหม่ (เปลี่ยนแค่ตัวเลข ไม่รีเฟรชจอ)
        else:
            st.error("ไม่พบโมเดล")
    else:
        with placeholder.container():
            st.warning("⏸️ ระบบหยุดทำงาน ติ๊กถูกที่เมนูด้านซ้ายเพื่อเริ่มการทำงาน Real-time")

elif app_mode == "📂 โหมด Test (Upload CSV)":
    st.info("โหมดทดสอบไฟล์ CSV ปิดอยู่ชั่วคราวเพื่อรันโหมด Live กรุณาใช้งานโหมด Live เพื่อดูความสมูทของระบบครับ")