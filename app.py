import streamlit as st
import streamlit.components.v1 as components
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
    page_icon="https://img5.pic.in.th/file/secure-sv1/706507857101af6d43e9fafd62b35bc6.png",
    layout="wide"
)

# ซ่อน UI ของ Streamlit
st.markdown("""
    <style>
        .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
        #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# แก้ไข Title
components.html(
    """<script>
        const targetTitle = 'UFP SENSE Dashboard';
        const titleEl = window.parent.document.querySelector('title');
        if (titleEl) titleEl.innerText = targetTitle;
    </script>""", height=0, width=0
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
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DATABASE_URL})
    except Exception as e: pass

def fetch_latest_firebase_data(limit=12):
    try:
        ref = db.reference(FIREBASE_NODE_NAME)
        data = ref.order_by_key().limit_to_last(limit).get()
        if data:
            records = [val for key, val in data.items()]
            return pd.DataFrame(records)
    except Exception as e: pass
    return pd.DataFrame()

# ==========================================
# [ส่วนที่ 1: คลาสโมเดล GRU]
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
    if os.path.exists(base_folder):
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
    if pm01 >= 20000: return "Hazardous air conditions detected. Please remain indoors and use air purifiers."
    elif pm01 >= 10000: return "Air quality is noticeably reduced. Turn on air purifiers."
    elif pm01 >= 1000: return "Air quality is acceptable. Sensitive individuals should monitor for discomfort."
    else: return "The air quality is excellent. Safe for normal activities."

# ==========================================
# [ส่วนที่ 2: ฟังก์ชันสร้างหน้าเว็บ (โหลดครั้งเดียว)]
# ==========================================
def render_main_ui():
    try:
        with open("index.html", "r", encoding="utf-8") as f: html_content = f.read()
        with open("style.css", "r", encoding="utf-8") as f: css_content = f.read()
        with open("script.js", "r", encoding="utf-8") as f: js_content = f.read()
        
        # 💡 [สำคัญมาก] แทรกสคริปต์เพื่อเปิดช่องทางให้ Python ยิงตัวเลขเข้ามาระหว่างทำงานได้
        magic_script = f"""
        <script>
            {js_content}
            window.parent.myDashboardWin = window;
            window.parent.myDashboardDoc = document;
        </script>
        """
        return f"<style>{css_content}</style>{html_content}{magic_script}"
    except FileNotFoundError:
        return "<h3 style='color:red;'>Error: ไม่พบไฟล์เว็บ</h3>"

# ฟังก์ชันนี้ใช้สำหรับ "ยิงตัวเลข" เข้าไปอัปเดตบนจอโดยไม่โหลดหน้าเว็บใหม่
def inject_data_to_ui(pm01, pm25, temp, humid, wind_dir, ai_text):
    c_cur = [int(pm01*0.8), int(pm01*1.05), int(pm01*0.95), int(pm01*1.1), int(pm01*0.9), pm01]
    c_hr = [int(pm01*1.2), int(pm01*1.1), int(pm01*1.0), int(pm01*0.9), pm01]
    c_day = [int(pm01*0.6), int(pm01*0.8), int(pm01*1.4), int(pm01*1.2), int(pm01*0.9), int(pm01*1.1), pm01]
    
    return f"""
    <script>
        var win = window.parent.myDashboardWin;
        var doc = window.parent.myDashboardDoc;
        
        if(win && doc) {{
            if(win.updateStatus) win.updateStatus({pm01}, "{ai_text}");
            if(win.updateWindDirection) win.updateWindDirection({wind_dir});
            if(win.updateCharts) win.updateCharts({c_cur}, {c_hr}, {c_day});
            
            if(doc.getElementById('val-pm25')) doc.getElementById('val-pm25').innerText = '{pm25}';
            if(doc.getElementById('val-temp')) doc.getElementById('val-temp').innerText = '{temp}';
            if(doc.getElementById('val-humid')) doc.getElementById('val-humid').innerText = '{humid}';
            if(doc.getElementById('modelNameDisplay')) doc.getElementById('modelNameDisplay').innerText = 'Live Predicted by GRU Model';
        }}
    </script>
    """

# ==========================================
# [ส่วนที่ 3: ระบบการทำงาน (Streamlit UI)]
# ==========================================
gru_model_data, device = load_models()

st.sidebar.title("⚙️ Control Panel")
app_mode = st.sidebar.radio("เลือกโหมด:", ("📡 โหมด Live (Firebase)", "📂 โหมด Test (Upload CSV)"))
st.sidebar.markdown("---")

# 1. แสดงหน้าเว็บของคุณทิ้งไว้ (แสดงแค่ครั้งเดียว จะไม่กระพริบอีกเลย)
ui_container = st.container()
with ui_container:
    components.html(render_main_ui(), height=1100, scrolling=True)

# 2. เตรียมช่องทางลับสำหรับยิงข้อมูล
injector_placeholder = st.empty()

if app_mode == "📡 โหมด Live (Firebase)":
    st.sidebar.markdown("**ระบบดึงข้อมูล Real-time อัตโนมัติ**")
    
    start_btn = st.sidebar.button("▶️ เริ่มรันระบบ Live (Start)")
    st.sidebar.markdown("*(หากต้องการหยุด ให้คลิกปุ่ม **Stop** ที่มุมขวาบนของจอ)*")

    if start_btn:
        st.sidebar.success("📡 ระบบกำลังทำงานและแสดงผลต่อเนื่อง...")
        if gru_model_data is not None:
            mod_i, prep_i, y_scaler_i, seq_len_i = gru_model_data
            
            # ลูปอมตะ: รับค่า -> ทำนาย -> ยิงขึ้นจอ -> วนกลับไปรับค่าใหม่
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
                        pm25 = round(last_row['Outdoor_PM2.5'], 2)
                        temp = round(last_row['Outdoor_Temperature'], 2)
                        humid = round(last_row['Outdoor_Humidity'], 2)
                        wind_dir = round(last_row['Wind_Dir'], 2)
                        ai_text = generate_ai_insight(pred_val)
                        
                        # ยิงข้อมูลตัวเลขใหม่เข้าไปในจอ (ไม่โหลดหน้าเว็บใหม่ ไม่ขาวแว๊บ)
                        with injector_placeholder:
                            components.html(inject_data_to_ui(pred_val, pm25, temp, humid, wind_dir, ai_text), height=0)
                
                # หน่วงเวลา 2 วินาที ก่อนวนลูปดึงข้อมูลรอบถัดไป
                time.sleep(2)
        else:
            st.error("โมเดลไม่พร้อมทำงาน")

elif app_mode == "📂 โหมด Test (Upload CSV)":
    st.sidebar.markdown("**อัปโหลดไฟล์เพื่อทดสอบ**")
    uploaded_file = st.sidebar.file_uploader("เลือกไฟล์ CSV", type=["csv"])

    if uploaded_file is not None and gru_model_data is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            mod_i, prep_i, y_scaler_i, seq_len_i = gru_model_data
            model_cols = ['Wind_Dir', 'Wind_Speed', 'Outdoor_Temperature', 'Outdoor_Humidity', 'Bar', 'Outdoor_PM2.5']
            
            if len(input_df) >= seq_len_i:
                X_proc_all = prep_i.transform(input_df[model_cols])
                seq_tensor = torch.FloatTensor(np.array([X_proc_all[i : i + seq_len_i] for i in range(len(X_proc_all) - seq_len_i + 1)])).to(device)
                
                mod_i.eval()
                with torch.no_grad():
                    out = mod_i(seq_tensor)
                
                raw_preds = y_scaler_i.inverse_transform(out.cpu().numpy())
                pred_val = int(raw_preds[-1][0] / 1000) 
                
                last_row = input_df.iloc[-1]
                pm25 = round(last_row['Outdoor_PM2.5'], 2)
                temp = round(last_row['Outdoor_Temperature'], 2)
                humid = round(last_row['Outdoor_Humidity'], 2)
                wind_dir = round(last_row['Wind_Dir'], 2)
                ai_text = generate_ai_insight(pred_val)
                
                # ยิงข้อมูลทดสอบเข้าไปในจอ
                with injector_placeholder:
                    components.html(inject_data_to_ui(pred_val, pm25, temp, humid, wind_dir, ai_text), height=0)
        except Exception as e:
            pass