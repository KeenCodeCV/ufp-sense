import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials, db

st.set_page_config(
    page_title="UFP SENSE Dashboard",
    page_icon="https://img5.pic.in.th/file/secure-sv1/706507857101af6d43e9fafd62b35bc6.png",
    layout="wide"
)

components.html(
    """
    <script>
        const targetTitle = 'UFP SENSE Dashboard';
        const titleEl = window.parent.document.querySelector('title');
        if (titleEl) {
            titleEl.innerText = targetTitle;
            const observer = new MutationObserver(() => {
                if (titleEl.innerText !== targetTitle) {
                    titleEl.innerText = targetTitle;
                }
            });
            observer.observe(titleEl, { childList: true, characterData: true, subtree: true });
        }
    </script>
    """,
    height=0,
    width=0,
)

# ==========================================
# [ส่วนที่ 0: ตั้งค่าการเชื่อมต่อ Firebase]
# ==========================================
FIREBASE_CREDENTIALS_FILE = "serviceAccountKey.json" 
FIREBASE_DATABASE_URL = "https://lab10-f138a-default-rtdb.asia-southeast1.firebasedatabase.app/" 
FIREBASE_NODE_NAME = "raw_sensor_data/history" # ⬅️ ชี้เป้าไปที่โหนดประวัติ

if not firebase_admin._apps:
    try:
        if "firebase" in st.secrets:
            cred_dict = dict(st.secrets["firebase"])
            cred = credentials.Certificate(cred_dict)
        elif os.path.exists(FIREBASE_CREDENTIALS_FILE):
            cred = credentials.Certificate(FIREBASE_CREDENTIALS_FILE)
        else:
            st.sidebar.error("❌ ไม่พบกุญแจเชื่อมต่อ Firebase ทั้งในไฟล์และ Secrets")
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

# ==========================================
# [ส่วนที่ 2: ฟังก์ชันโหลดโมเดล GRU]
# ==========================================
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_folder = "models"
    model_data = None
    
    if not os.path.exists(base_folder):
        st.error(f"❌ ไม่พบโฟลเดอร์ '{base_folder}' ในระบบ โปรดสร้างโฟลเดอร์และใส่ไฟล์โมเดล")
        return model_data, device

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
        else:
            st.warning(f"⚠️ หาไฟล์ GRU ไม่พบ: {path_gru}")
    except Exception as e: 
        st.error(f"GRU Error: {e}")
    
    return model_data, device

# ==========================================
# [ส่วนฟังก์ชัน สร้าง AI Insight อัจฉริยะ]
# ==========================================
def generate_ai_insight(pm01, pm25, wind_speed, temp, humid):
    if pm01 >= 20000:
        return "Hazardous air conditions detected. Please remain indoors, keep all windows closed, and use high-efficiency air purifiers immediately."
    elif pm01 >= 10000:
        return "Air quality is noticeably reduced. It is highly recommended to turn on air purifiers and minimize outdoor activities."
    elif pm01 >= 1000:
        return "Air quality is acceptable. However, sensitive individuals should monitor for any discomfort and consider limiting prolonged outdoor exertion."
    else:
        return "The air quality is excellent. The environment is safe, making it a great time for normal activities and natural ventilation."

# ==========================================
# [ส่วนที่ 3: ฟังก์ชันอ่านไฟล์เว็บและ Inject ค่า]
# ==========================================
def render_web_interface(pm01_val, pm25_val, temp_val, humid_val, wind_val, wind_speed, is_active):
    try:
        with open("index.html", "r", encoding="utf-8") as f: html_content = f.read()
        with open("style.css", "r", encoding="utf-8") as f: css_content = f.read()
        with open("script.js", "r", encoding="utf-8") as f: js_content = f.read()
    except FileNotFoundError:
        return "<h3 style='color:red; text-align:center;'>Error: ไม่พบไฟล์ HTML/CSS/JS</h3>"

    if not is_active:
        injection_script = f"<script>{js_content}\n setTimeout(function() {{ if(window.resetSystem) window.resetSystem(); }}, 500);</script>"
    else:
        ai_text = generate_ai_insight(pm01_val, pm25_val, wind_speed, temp_val, humid_val)
        chart_current = [int(pm01_val*0.8), int(pm01_val*1.05), int(pm01_val*0.95), int(pm01_val*1.1), int(pm01_val*0.9), pm01_val]
        chart_hour = [int(pm01_val*1.2), int(pm01_val*1.1), int(pm01_val*1.0), int(pm01_val*0.9), pm01_val]
        chart_day = [int(pm01_val*0.6), int(pm01_val*0.8), int(pm01_val*1.4), int(pm01_val*1.2), int(pm01_val*0.9), int(pm01_val*1.1), pm01_val]
        
        injection_script = f"""
        <script>
            {js_content} 
            setTimeout(function() {{
                if(window.updateStatus) window.updateStatus({pm01_val}, "{ai_text}");
                if(window.updateWindDirection) window.updateWindDirection({wind_val});
                if(window.updateCharts) window.updateCharts({chart_current}, {chart_hour}, {chart_day});
                
                var pm25Elem = document.getElementById('val-pm25');
                var tempElem = document.getElementById('val-temp');
                var humidElem = document.getElementById('val-humid');
                var modelElem = document.getElementById('modelNameDisplay');
                
                if(pm25Elem) pm25Elem.innerText = '{pm25_val}';
                if(tempElem) tempElem.innerText = '{temp_val}';
                if(humidElem) humidElem.innerText = '{humid_val}';
                if(modelElem) modelElem.innerText = 'Live Predicted by GRU Model';
            }}, 500);
        </script>
        """

    return f"<style>{css_content}</style>{html_content}{injection_script}"

# ==========================================
# [ส่วนที่ 4: Streamlit UI]
# ==========================================
st.markdown("""
    <style>
        .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
        #MainMenu {visibility: hidden;} 
        footer {visibility: hidden;}
        [data-testid="stAppDeployButton"] {display: none !important;}
        .viewerBadge_container {display: none !important;}
        .viewerBadge_link {display: none !important;}
        div[data-testid="stAlert"] { margin-top: -15px !important; padding: 10px !important; }
    </style>
""", unsafe_allow_html=True)

gru_model_data, device = load_models()

st.sidebar.title("⚙️ Control Panel")
if not gru_model_data:
    st.sidebar.error("❌ ไม่พบโมเดล GRU! กรุณาตรวจสอบโฟลเดอร์ 'models'")

app_mode = st.sidebar.radio(
    "เลือกโหมดการทำงาน:",
    ("📡 โหมด Live (Firebase)", "📂 โหมด Test (Upload CSV)")
)

st.sidebar.markdown("---")

# ==========================================
# โหมดที่ 1: Live (Firebase)
# ==========================================
if app_mode == "📡 โหมด Live (Firebase)":
    st.sidebar.markdown("**ดึงข้อมูลสดจากเซ็นเซอร์**")
    fetch_button = st.sidebar.button("🔄 ดึงข้อมูลล่าสุด (Fetch Data)", type="primary")

    if fetch_button:
        if gru_model_data is not None:
            with st.spinner('📡 กำลังดึงข้อมูลจาก Firebase...'):
                mod_i, prep_i, y_scaler_i, seq_len_i = gru_model_data
                input_df = fetch_latest_firebase_data(limit=seq_len_i)
                
                if not input_df.empty:
                    # 💡 แปลงชื่อคอลัมน์จาก API ให้ตรงกับ Model
                    rename_map = {
                        'wind_dir': 'Wind_Dir',
                        'wind_speed': 'Wind_Speed',
                        'outdoor_temp': 'Outdoor_Temperature',
                        'outdoor_hum': 'Outdoor_Humidity',
                        'bar': 'Bar',
                        'outdoor_pm25': 'Outdoor_PM2.5'
                    }
                    input_df = input_df.rename(columns=rename_map)

                    model_cols = ['Wind_Dir', 'Wind_Speed', 'Outdoor_Temperature', 'Outdoor_Humidity', 'Bar', 'Outdoor_PM2.5']
                    missing_cols = [c for c in model_cols if c not in input_df.columns]
                    
                    if missing_cols:
                        st.error(f"❌ ข้อมูลใน Firebase ขาดตัวแปรเหล่านี้: {', '.join(missing_cols)}")
                        components.html(render_web_interface(0, 0, 0, 0, 0, 0, False), height=1100, scrolling=True)
                    elif len(input_df) >= seq_len_i:
                        X_proc_all = prep_i.transform(input_df[model_cols])
                        seq_tensor = torch.FloatTensor(X_proc_all).unsqueeze(0).to(device) 
                        
                        mod_i.eval()
                        with torch.no_grad():
                            out = mod_i(seq_tensor)
                        
                        raw_pred = y_scaler_i.inverse_transform(out.cpu().numpy())
                        pred_val = int(raw_pred[0][0] / 1000) 
                        
                        last_row = input_df.iloc[-1]
                        pm25_disp = round(last_row['Outdoor_PM2.5'], 2)
                        temp_disp = round(last_row['Outdoor_Temperature'], 2)
                        humid_disp = round(last_row['Outdoor_Humidity'], 2)
                        wind_spd = round(last_row['Wind_Speed'], 2)
                        wind_disp = round(last_row['Wind_Dir'], 2)
                        
                        st.sidebar.success("✅ อัปเดตข้อมูลสำเร็จ!")
                        html_content = render_web_interface(pred_val, pm25_disp, temp_disp, humid_disp, wind_disp, wind_spd, True)
                        components.html(html_content, height=1100, scrolling=True)
                    else:
                        st.warning(f"⚠️ ข้อมูลใน Firebase มีไม่ถึง {seq_len_i} แถว (มี {len(input_df)} แถว) รอเซ็นเซอร์ส่งข้อมูลอีกนิดนะครับ")
                        components.html(render_web_interface(0, 0, 0, 0, 0, 0, False), height=1100, scrolling=True)
                else:
                    st.error("❌ ไม่พบข้อมูลใน Firebase หรือหา Node history ไม่เจอ")
                    components.html(render_web_interface(0, 0, 0, 0, 0, 0, False), height=1100, scrolling=True)
        else:
            components.html(render_web_interface(0, 0, 0, 0, 0, 0, False), height=1100, scrolling=True)
    else:
        components.html(render_web_interface(0, 0, 0, 0, 0, 0, False), height=1100, scrolling=True)

# ==========================================
# โหมดที่ 2: Test (Upload CSV)
# ==========================================
elif app_mode == "📂 โหมด Test (Upload CSV)":
    st.sidebar.markdown("**อัปโหลดไฟล์เพื่อทดสอบ**")
    uploaded_file = st.sidebar.file_uploader("เลือกไฟล์ CSV", type=["csv"])

    if uploaded_file is not None:
        if gru_model_data is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                original_cols = input_df.columns.tolist() 
                
                model_cols = ['Wind_Dir', 'Wind_Speed', 'Outdoor_Temperature', 'Outdoor_Humidity', 'Bar', 'Outdoor_PM2.5']
                missing_cols = [c for c in model_cols if c not in input_df.columns]
                
                if missing_cols:
                     st.error(f"❌ ไฟล์ CSV ขาดคอลัมน์พื้นฐาน: {', '.join(missing_cols)}")
                     st.markdown("<br><br>", unsafe_allow_html=True)
                     components.html(render_web_interface(0, 0, 0, 0, 0, 0, False), height=1100, scrolling=True)
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
                    
                    st.sidebar.markdown("---")
                    st.sidebar.success("✅ ประมวลผลโมเดล GRU เสร็จสิ้น!")
                    st.sidebar.download_button(
                        label="📥 ดาวน์โหลดไฟล์ผลลัพธ์ (CSV)",
                        data=csv_data,
                        file_name=f"predicted_GRU_{uploaded_file.name}",
                        mime="text/csv",
                    )

                    last_pred = download_df.iloc[-1]['Predict_GRU_Indoor_PC0.1']
                    pred_val = int(last_pred) if pd.notna(last_pred) else 0

                    last_row = input_df.iloc[-1]
                    pm25_disp = round(last_row['Outdoor_PM2.5'], 2)
                    temp_disp = round(last_row['Outdoor_Temperature'], 2)
                    humid_disp = round(last_row['Outdoor_Humidity'], 2)
                    wind_spd = round(last_row['Wind_Speed'], 2)
                    wind_disp = round(last_row['Wind_Dir'], 2)
                    
                    html_content = render_web_interface(pred_val, pm25_disp, temp_disp, humid_disp, wind_disp, wind_spd, True)
                    components.html(html_content, height=1100, scrolling=True)
                    
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์หรือประมวลผล: {e}")
                st.markdown("<br><br>", unsafe_allow_html=True)
                components.html(render_web_interface(0, 0, 0, 0, 0, 0, False), height=1100, scrolling=True)
        else:
            components.html(render_web_interface(0, 0, 0, 0, 0, 0, False), height=1100, scrolling=True)
    else:
        components.html(render_web_interface(0, 0, 0, 0, 0, 0, False), height=1100, scrolling=True)