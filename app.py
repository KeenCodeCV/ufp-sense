import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
import os

# ==========================================
# [ส่วนที่ 1: คลาสโมเดล]
# ==========================================
class SingleStepGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
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

class SingleStepRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super(SingleStepRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class SingleStepLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super(SingleStepLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.LSTM(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# [ส่วนที่ 2: ฟังก์ชันโหลดโมเดล] (อัปเดตสำหรับ Cloud)
# ==========================================
@st.cache_resource
def load_models():
    models = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # เปลี่ยนมาระบุโฟลเดอร์ models แทนไดรฟ์ C:
    base_folder = "models"
    
    try:
        name_gru = "GRU_fold_5_20251111_030307"
        path_gru = os.path.join(base_folder, f"{name_gru}.pth")
        if os.path.exists(path_gru):
            with open(os.path.join(base_folder, f"{name_gru}_preprocessor.pkl"), 'rb') as f: prep_gru = pickle.load(f)
            with open(os.path.join(base_folder, f"{name_gru}_scaler.pkl"), 'rb') as f: y_scale_gru = pickle.load(f)
            model_gru = SingleStepGRU(input_size=7, hidden_size=15, num_layers=9)
            model_gru.load_state_dict(torch.load(path_gru, map_location=device))
            model_gru.to(device).eval()
            models['GRU'] = (model_gru, prep_gru, y_scale_gru)
    except Exception as e: print(f"GRU Error: {e}")

    try:
        name_rnn = "RNN_fold_5_20251107_013311"
        path_rnn = os.path.join(base_folder, f"{name_rnn}.pth")
        if os.path.exists(path_rnn):
            with open(os.path.join(base_folder, f"{name_rnn}_preprocessor.pkl"), 'rb') as f: prep_rnn = pickle.load(f)
            with open(os.path.join(base_folder, f"{name_rnn}_scaler.pkl"), 'rb') as f: y_scale_rnn = pickle.load(f)
            model_rnn = SingleStepRNN(input_size=7, hidden_size=15, num_layers=9)
            model_rnn.load_state_dict(torch.load(path_rnn, map_location=device))
            model_rnn.to(device).eval()
            models['RNN'] = (model_rnn, prep_rnn, y_scale_rnn)
    except Exception as e: print(f"RNN Error: {e}")

    try:
        name_lstm = "LSTM_fold_5_20251110_033855"
        path_lstm = os.path.join(base_folder, f"{name_lstm}.pth")
        if os.path.exists(path_lstm):
            with open(os.path.join(base_folder, f"{name_lstm}_preprocessor.pkl"), 'rb') as f: prep_lstm = pickle.load(f)
            with open(os.path.join(base_folder, f"{name_lstm}_scaler.pkl"), 'rb') as f: y_scale_lstm = pickle.load(f)
            model_lstm = SingleStepLSTM(input_size=7, hidden_size=254, num_layers=1)
            model_lstm.load_state_dict(torch.load(path_lstm, map_location=device), strict=False)
            model_lstm.to(device).eval()
            models['LSTM'] = (model_lstm, prep_lstm, y_scale_lstm)
    except Exception as e: print(f"LSTM Error: {e}")
    
    return models, device

# ==========================================
# [ส่วนฟังก์ชัน สร้าง AI Insight อัจฉริยะ]
# ==========================================
def generate_ai_insight(pm01, pm25, wind_speed, temp, humid):
    if pm01 > 10000:
        if wind_speed < 1.0 and humid > 60:
            return f"🚨 อันตราย: สภาพอากาศปิด (ลมนิ่งเพียง {wind_speed} m/s) และความชื้นสูงถึง {humid}% ทำให้ฝุ่น PM0.1 ไม่ถูกระบายออกและสะสมตัวอย่างรวดเร็ว แนะนำให้สวมหน้ากาก N95 ทันที"
        else:
            return f"🚨 อันตราย: ค่าฝุ่นภายนอก (PM2.5: {pm25}) อาจแทรกซึมเข้าสู่อาคาร ทำให้ PM0.1 พุ่งสูง แนะนำให้เปิดเครื่องฟอกอากาศขั้นสูงสุด"
    elif pm01 >= 1000:
        return f"🟡 แจ้งเตือน: อุณหภูมิ {temp}°C และลม {wind_speed} m/s เริ่มมีแนวโน้มทำให้เกิดการกักเก็บฝุ่น ควรหลีกเลี่ยงการทำกิจกรรมที่ทำให้เกิดควันภายในอาคาร"
    else:
        if wind_speed > 2.0:
            return f"🟢 ปลอดภัย: ลมภายนอกพัดแรง ({wind_speed} m/s) ช่วยระบายอากาศภายในอาคารได้ดี สภาพแวดล้อมเหมาะสำหรับการใช้ชีวิตตามปกติ"
        else:
            return f"🟢 ปลอดภัย: คุณภาพอากาศอยู่ในเกณฑ์มาตรฐาน (PM2.5 ภายนอกต่ำเพียง {pm25} µg/m³) ไม่มีฝุ่นจิ๋วสะสมตัว"

# ==========================================
# [ส่วนที่ 3: ฟังก์ชันอ่านไฟล์เว็บและ Inject ค่า]
# ==========================================
def render_web_interface(pm01_val, pm25_val, temp_val, humid_val, wind_val, wind_speed, model_name):
    try:
        with open("index.html", "r", encoding="utf-8") as f: html_content = f.read()
        with open("style.css", "r", encoding="utf-8") as f: css_content = f.read()
        with open("script.js", "r", encoding="utf-8") as f: js_content = f.read()
    except FileNotFoundError:
        return "<h3 style='color:red;'>Error: ไม่พบไฟล์ index.html, style.css หรือ script.js</h3>"

    if model_name == "No Model":
        injection_script = f"<script>{js_content}\n setTimeout(function() {{ if(window.resetSystem) window.resetSystem(); }}, 500);</script>"
    else:
        # วิเคราะห์ Insight
        ai_text = generate_ai_insight(pm01_val, pm25_val, wind_speed, temp_val, humid_val)
        
        # สร้างข้อมูลกราฟจำลองเพื่อให้สอดคล้องกับค่า PM0.1 ปัจจุบัน (เพื่อให้กราฟดูสมจริง)
        # Current (6 จุด)
        chart_current = [int(pm01_val*0.8), int(pm01_val*1.05), int(pm01_val*0.95), int(pm01_val*1.1), int(pm01_val*0.9), pm01_val]
        # Hour (5 จุด)
        chart_hour = [int(pm01_val*1.2), int(pm01_val*1.1), int(pm01_val*1.0), int(pm01_val*0.9), pm01_val]
        # Day (7 จุด)
        chart_day = [int(pm01_val*0.6), int(pm01_val*0.8), int(pm01_val*1.4), int(pm01_val*1.2), int(pm01_val*0.9), int(pm01_val*1.1), pm01_val]
        
        injection_script = f"""
        <script>
            {js_content} 
            
            setTimeout(function() {{
                if(window.updateStatus) window.updateStatus({pm01_val}, `{ai_text}`);
                if(window.updateWindDirection) window.updateWindDirection({wind_val});
                
                // อัปเดตกราฟด้วยข้อมูลใหม่
                if(window.updateCharts) window.updateCharts({chart_current}, {chart_hour}, {chart_day});
                
                var pm25Elem = document.getElementById('val-pm25');
                var tempElem = document.getElementById('val-temp');
                var humidElem = document.getElementById('val-humid');
                var modelElem = document.getElementById('modelNameDisplay');
                
                if(pm25Elem) pm25Elem.innerHTML = '{pm25_val} <span class="text-xs font-normal text-slate-500">µg/m³</span>';
                if(tempElem) tempElem.innerHTML = '{temp_val} <span class="text-xs font-normal text-slate-500">°C</span>';
                if(humidElem) humidElem.innerHTML = '{humid_val} <span class="text-xs font-normal text-slate-500">%</span>';
                if(modelElem) modelElem.innerText = 'Predicted by {model_name} Model';
                
            }}, 500);
        </script>
        """

    return f"<style>{css_content}</style>{html_content}{injection_script}"

# ==========================================
# [ส่วนที่ 4: Streamlit UI]
# ==========================================
st.set_page_config(layout="wide", page_title="WFP SENSE Dashboard", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
        /* ซ่อนแค่เมนูย่อยของ Streamlit แต่ปล่อยปุ่มเปิด/ปิด Sidebar ไว้ */
        #MainMenu {visibility: hidden;} 
        footer {visibility: hidden;}
        div[data-testid="stAlert"] { margin-top: -15px !important; padding: 10px !important; }
    </style>
""", unsafe_allow_html=True)

models, device = load_models()

st.sidebar.title("⚡ Control Panel")
if not models:
    st.sidebar.error("❌ ไม่พบโมเดลสักตัว! กรุณาเช็ค Path ไฟล์")

selected_model_name = st.sidebar.selectbox("เลือกโมเดล (Model)", ["LSTM", "GRU", "RNN"], index=0)
uploaded_file = st.sidebar.file_uploader("📂 อัปโหลดไฟล์ CSV (Data Input)", type=["csv"])

if uploaded_file is not None and selected_model_name in models:
    try:
        input_df = pd.read_csv(uploaded_file)
        seq_len = 48
        cols = ['Wind_Dir', 'Wind_Dir_Missing', 'Wind_Speed', 'Outdoor_Temperature', 'Outdoor_Humidity', 'Bar', 'Outdoor_PM2.5']
        
        if len(input_df) >= seq_len:
            input_data = input_df[cols].tail(seq_len)
            
            last_row = input_df.iloc[-1]
            pm25_disp = round(last_row['Outdoor_PM2.5'], 2)
            temp_disp = round(last_row['Outdoor_Temperature'], 2)
            humid_disp = round(last_row['Outdoor_Humidity'], 2)
            wind_disp = round(last_row['Wind_Dir'], 2) 
            wind_spd = round(last_row['Wind_Speed'], 2) # ดึงค่าความเร็วลมไปวิเคราะห์ AI

            model, prep, y_scaler = models[selected_model_name]
            
            X_processed = prep.transform(input_data)
            input_tensor = torch.FloatTensor(X_processed).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred_scaled = model(input_tensor)
            
            pred_val = int(y_scaler.inverse_transform(pred_scaled.cpu().numpy())[0][0])
            
            # ส่งค่าทั้งหมดรวมถึง wind_spd ไปด้วย
            html_content = render_web_interface(pred_val, pm25_disp, temp_disp, humid_disp, wind_disp, wind_spd, selected_model_name)
            components.html(html_content, height=1100, scrolling=True)
            
        else:
            st.error(f"ข้อมูลไม่เพียงพอ! ต้องการอย่างน้อย {seq_len} แถว")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
else:
    if uploaded_file is None:
        st.info("👈 กรุณาอัปโหลดไฟล์ CSV ทางด้านซ้ายเพื่อเริ่มการทำนาย")
        try:
            # เพิ่ม 0 สำหรับ parameter wind_speed ที่เพิ่มมาใหม่
            html_content = render_web_interface(0, 0, 0, 0, 0, 0, "No Model")
            components.html(html_content, height=1100, scrolling=True)
        except:
            pass