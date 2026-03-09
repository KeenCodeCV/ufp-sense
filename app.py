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
# [ส่วนที่ 2: ฟังก์ชันสร้างหน้าเว็บและอัปเดตกราฟ]
# ==========================================
def render_main_ui():
    try:
        with open("index.html", "r", encoding="utf-8") as f: html_content = f.read()
        with open("style.css", "r", encoding="utf-8") as f: css_content = f.read()
        with open("script.js", "r", encoding="utf-8") as f: js_content = f.read()
        
        magic_script = f"""
        <script>
            {js_content}
            window.parent.myDashboardWin = window;
            window.parent.myDashboardDoc = document;
        </script>
        """
        return f"<style>{css_content}</style>{html_content}{magic_script}"
    except FileNotFoundError:
        return "<h3 style='color:red;'>Error: ไม่พบไฟล์เว็บ HTML/CSS/JS</h3>"

# 💡 [เปลี่ยนใหม่] รับค่าอาร์เรย์กราฟของจริงเข้ามา (c_cur, c_hr, c_day) แทนการคำนวณหลอกๆ
def inject_data_to_ui(pm01, pm25, temp, humid, wind_dir, ai_text, c_cur, c_hr, c_day):
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

# 💡 ฟังก์ชันใหม่: เอาไว้คำนวณหาค่าเฉลี่ยของกราฟจากประวัติที่มีอยู่จริง
def calculate_trend(data_list, points_needed):
    if not data_list: return [0] * points_needed
    # ถ้าข้อมูลยังมีไม่พอ ให้ก๊อปปี้ค่าแรกสุดมาเติมให้เต็มกราฟไปก่อน
    if len(data_list) < points_needed:
        return [int(data_list[0])] * (points_needed - len(data_list)) + [int(x) for x in data_list]
    
    # หั่นข้อมูลออกเป็นท่อนๆ เท่าๆ กัน แล้วหาค่าเฉลี่ย
    chunk_size = len(data_list) // points_needed
    data_to_use = data_list[-(points_needed * chunk_size):]
    
    result = []
    for i in range(points_needed):
        chunk = data_to_use[i * chunk_size : (i + 1) * chunk_size]
        result.append(int(sum(chunk) / len(chunk)))
    return result

# ==========================================
# [ส่วนที่ 3: ระบบการทำงาน (Streamlit UI)]
# ==========================================
gru_model_data, device = load_models()

st.sidebar.title("⚙️ Control Panel")
app_mode = st.sidebar.radio("เลือกโหมด:", ("📡 โหมด Live (Firebase)", "📂 โหมด Test (Upload CSV)"))

st.sidebar.markdown("---")
st.sidebar.markdown("**🛠️ เครื่องมือทดสอบหน้าจอ**")
col1, col2 = st.sidebar.columns(2)
btn_test = col1.button("▶️ Test", use_container_width=True)
btn_reset = col2.button("🔄 Reset", use_container_width=True)
st.sidebar.markdown("---")

ui_container = st.container()
with ui_container:
    components.html(render_main_ui(), height=1100, scrolling=True)

injector_placeholder = st.empty()

if btn_test:
    with injector_placeholder:
        components.html("""<script>if(window.parent.myDashboardWin && window.parent.myDashboardWin.startTestMode) window.parent.myDashboardWin.startTestMode();</script>""", height=0)

if btn_reset:
    with injector_placeholder:
        components.html("""<script>if(window.parent.myDashboardWin && window.parent.myDashboardWin.resetSystem) window.parent.myDashboardWin.resetSystem();</script>""", height=0)

# ==========================================
# โหมด Live (Firebase)
# ==========================================
if app_mode == "📡 โหมด Live (Firebase)":
    
    # 💡 1. สร้าง "ความจำ" ให้เว็บจำได้ว่าเราเคยกดปุ่ม Start ไปหรือยัง
    if 'is_live_running' not in st.session_state:
        st.session_state.is_live_running = False

    # สร้างปุ่ม Start และ Stop ควบคู่กัน
    col1, col2 = st.sidebar.columns(2)
    start_btn = col1.button("🚀 Start Live", type="primary", use_container_width=True)
    stop_btn = col2.button("🛑 Stop", use_container_width=True)

    if start_btn:
        st.session_state.is_live_running = True
    if stop_btn:
        st.session_state.is_live_running = False

    # 💡 2. ถ้าระบบจำได้ว่า "กำลังรันอยู่" ก็ให้ทำงานต่อไปโดยไม่ต้องรอคนมากดซ้ำ
    if st.session_state.is_live_running:
        st.sidebar.success("📡 ระบบกำลังทำงานและแสดงผลต่อเนื่อง... (รันอัตโนมัติ)")
        
        if gru_model_data is not None:
            mod_i, prep_i, y_scaler_i, seq_len_i = gru_model_data
            
            # 💡 3. เก็บถังประวัติข้อมูลไว้ใน ความจำของเว็บ (จะได้ไม่ต้องโหลดใหม่ทุกรอบ)
            if 'history_pm01' not in st.session_state:
                st.session_state.history_pm01 = []
                try:
                    ref_hist = db.reference(FIREBASE_NODE_NAME).order_by_key().limit_to_last(2000).get()
                    if ref_hist:
                        # หาร 1000 เพื่อแก้ปัญหากราฟสีแดงทะลุหลอดเรียบร้อยแล้ว!
                        st.session_state.history_pm01 = [int(float(v.get('indoor_pc01_raw', 0)) / 1000) for k, v in ref_hist.items() if 'indoor_pc01_raw' in v]
                except: pass
            
            # 💡 4. ดึงข้อมูล 1 รอบ (ลบ while True ทิ้งไปเลย!)
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
                    
                    st.session_state.history_pm01.append(pred_val)
                    if len(st.session_state.history_pm01) > 28800:
                        st.session_state.history_pm01.pop(0)
                    
                    c_cur = calculate_trend(st.session_state.history_pm01[-6:], 6)
                    c_hr = calculate_trend(st.session_state.history_pm01[-1200:], 5) 
                    c_day = calculate_trend(st.session_state.history_pm01[-28800:], 7) 
                    
                    # 💡 บังคับให้จุดสุดท้าย (Now) ของกราฟเท่ากับค่าปัจจุบันเป๊ะๆ
                    if len(c_hr) > 0: c_hr[-1] = pred_val
                    if len(c_day) > 0: c_day[-1] = pred_val 
                    
                    last_row = input_df.iloc[-1]
                    pm25 = round(last_row['Outdoor_PM2.5'], 2)
                    temp = round(last_row['Outdoor_Temperature'], 2)
                    humid = round(last_row['Outdoor_Humidity'], 2)
                    wind_dir = round(last_row['Wind_Dir'], 2)
                    ai_text = generate_ai_insight(pred_val)
                    
                    with injector_placeholder:
                        components.html(inject_data_to_ui(pred_val, pm25, temp, humid, wind_dir, ai_text, c_cur, c_hr, c_day), height=0)
            
            # 💡 5. พระเอกของงานนี้: ให้มันรอ 2 วินาที แล้วสั่ง "รีเฟรชตัวเองจากภายใน"
            time.sleep(1)
            st.rerun() 
            
        else:
            st.sidebar.error("โมเดลไม่พร้อมทำงาน")

# ==========================================
# โหมด Test (Upload CSV) 
# ==========================================
elif app_mode == "📂 โหมด Test (Upload CSV)":
    uploaded_file = st.sidebar.file_uploader("เลือกไฟล์ CSV", type=["csv"])

    if uploaded_file is not None:
        if gru_model_data is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                original_cols = input_df.columns.tolist() 
                
                model_cols = ['Wind_Dir', 'Wind_Speed', 'Outdoor_Temperature', 'Outdoor_Humidity', 'Bar', 'Outdoor_PM2.5']
                missing_cols = [c for c in model_cols if c not in input_df.columns]
                
                if missing_cols:
                     st.sidebar.error(f"❌ ไฟล์ CSV ขาดคอลัมน์: {', '.join(missing_cols)}")
                else:
                    mod_i, prep_i, y_scaler_i, seq_len_i = gru_model_data
                    download_df = input_df[original_cols].copy() 
                    
                    if len(input_df) >= seq_len_i:
                        X_proc_all = prep_i.transform(input_df[model_cols])
                        sequences = []
                        for i in range(len(X_proc_all) - seq_len_i + 1):
                            sequences.append(X_proc_all[i : i + seq_len_i])
                        
                        seq_tensor = torch.FloatTensor(np.array(sequences)).to(device)
                        
                        preds_list = []
                        chunk_size = 256
                        mod_i.eval()
                        with torch.no_grad():
                            for i in range(0, len(seq_tensor), chunk_size):
                                chunk = seq_tensor[i:i+chunk_size]
                                out = mod_i(chunk)
                                preds_list.append(out.cpu().numpy())
                        
                        if preds_list:
                            preds_scaled = np.vstack(preds_list)
                            raw_preds = y_scaler_i.inverse_transform(preds_scaled)
                            preds_final = (raw_preds / 1000).flatten().astype(int)
                        else:
                            preds_final = np.array([])
                        
                        full_predictions = [None] * (seq_len_i - 1) + preds_final.tolist()
                    else:
                        full_predictions = [None] * len(input_df)
                        
                    download_df['Predict_GRU_Indoor_PC0.1'] = full_predictions
                    csv_data = download_df.to_csv(index=False).encode('utf-8-sig') 
                    
                    st.sidebar.success("✅ ประมวลผลเสร็จสิ้น!")
                    st.sidebar.download_button(
                        label="📥 ดาวน์โหลดไฟล์ผลลัพธ์ (CSV)",
                        data=csv_data,
                        file_name=f"predicted_GRU_{uploaded_file.name}",
                        mime="text/csv",
                        use_container_width=True
                    )

                    # 💡 กราฟในโหมด CSV ก็คำนวณจากค่าจริงในไฟล์ CSV ด้วย!
                    csv_preds = [int(p) for p in full_predictions if pd.notna(p)]
                    c_cur = calculate_trend(csv_preds[-6:], 6) if csv_preds else [0]*6
                    c_hr = calculate_trend(csv_preds[-1200:], 5) if csv_preds else [0]*5
                    c_day = calculate_trend(csv_preds[-28800:], 7) if csv_preds else [0]*7

                    last_pred = download_df.iloc[-1]['Predict_GRU_Indoor_PC0.1']
                    pred_val = int(last_pred) if pd.notna(last_pred) else 0
                    
                    # 💡 บังคับให้จุดสุดท้าย (Now) ของกราฟเท่ากับค่าปัจจุบันเป๊ะๆ (โหมด CSV)
                    if len(c_hr) > 0: c_hr[-1] = pred_val
                    if len(c_day) > 0: c_day[-1] = pred_val

                    last_row = input_df.iloc[-1]
                    pm25 = round(last_row['Outdoor_PM2.5'], 2)
                    temp = round(last_row['Outdoor_Temperature'], 2)
                    humid = round(last_row['Outdoor_Humidity'], 2)
                    wind_dir = round(last_row['Wind_Dir'], 2)
                    ai_text = generate_ai_insight(pred_val)
                    
                    with injector_placeholder:
                        components.html(inject_data_to_ui(pred_val, pm25, temp, humid, wind_dir, ai_text, c_cur, c_hr, c_day), height=0)
                        
            except Exception as e:
                st.sidebar.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์หรือประมวลผล: {e}")
        else:
            st.sidebar.error("❌ โมเดลไม่พร้อมทำงาน")