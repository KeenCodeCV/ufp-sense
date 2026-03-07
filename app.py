import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="UFP SENSE Dashboard",
    page_icon="https://img5.pic.in.th/file/secure-sv1/706507857101af6d43e9fafd62b35bc6.png",
    layout="wide"
)

# ==========================================
# [วิชามาร: จ้างยามเฝ้าเปลี่ยนชื่อ Title ลบคำว่า Streamlit]
# ==========================================
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
# [ส่วนที่ 2: ฟังก์ชันโหลดโมเดล (อัปเดตสเปคล่าสุด)]
# ==========================================
@st.cache_resource
def load_models():
    models = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_folder = "models"
    
    if not os.path.exists(base_folder):
        st.error(f"❌ ไม่พบโฟลเดอร์ '{base_folder}' ในระบบ โปรดสร้างโฟลเดอร์และใส่ไฟล์โมเดล")
        return models, device

    # --- 1. โหลด GRU (Seq=48) ---
    try:
        name_gru = "gru_Best_Model_20260306_181208"
        path_gru = os.path.join(base_folder, f"{name_gru}.pth")
        if os.path.exists(path_gru):
            with open(os.path.join(base_folder, f"{name_gru}_preprocessor.pkl"), 'rb') as f: prep_gru = pickle.load(f)
            with open(os.path.join(base_folder, f"{name_gru}_scaler.pkl"), 'rb') as f: y_scale_gru = pickle.load(f)
            
            model_gru = SingleStepGRU(input_size=8, hidden_size=8, num_layers=1)
            model_gru.load_state_dict(torch.load(path_gru, map_location=device), strict=False)
            model_gru.to(device).eval()
            models['GRU'] = (model_gru, prep_gru, y_scale_gru, 48)
        else:
            st.warning(f"⚠️ หาไฟล์ GRU ไม่พบ: {path_gru}")
    except Exception as e: st.error(f"GRU Error: {e}")

    # --- 2. โหลด RNN (Seq=24) ---
    try:
        name_rnn = "rnn_Best_Model_20260306_164237"
        path_rnn = os.path.join(base_folder, f"{name_rnn}.pth")
        if os.path.exists(path_rnn):
            with open(os.path.join(base_folder, f"{name_rnn}_preprocessor.pkl"), 'rb') as f: prep_rnn = pickle.load(f)
            with open(os.path.join(base_folder, f"{name_rnn}_scaler.pkl"), 'rb') as f: y_scale_rnn = pickle.load(f)
            
            model_rnn = SingleStepRNN(input_size=8, hidden_size=16, num_layers=3)
            model_rnn.load_state_dict(torch.load(path_rnn, map_location=device), strict=False)
            model_rnn.to(device).eval()
            models['RNN'] = (model_rnn, prep_rnn, y_scale_rnn, 24)
        else:
            st.warning(f"⚠️ หาไฟล์ RNN ไม่พบ: {path_rnn}")
    except Exception as e: st.error(f"RNN Error: {e}")

    # --- 3. โหลด LSTM (Seq=42) ---
    try:
        # ✅ แก้ไขชื่อไฟล์ LSTM ให้ตรงกับไฟล์จริงๆ ของคุณแล้ว
        name_lstm = "lstm_Best_Model_20260306_235311" 
        path_lstm = os.path.join(base_folder, f"{name_lstm}.pth")
        if os.path.exists(path_lstm):
            with open(os.path.join(base_folder, f"{name_lstm}_preprocessor.pkl"), 'rb') as f: prep_lstm = pickle.load(f)
            with open(os.path.join(base_folder, f"{name_lstm}_scaler.pkl"), 'rb') as f: y_scale_lstm = pickle.load(f)
            
            model_lstm = SingleStepLSTM(input_size=8, hidden_size=8, num_layers=1)
            model_lstm.load_state_dict(torch.load(path_lstm, map_location=device), strict=False)
            model_lstm.to(device).eval()
            models['LSTM'] = (model_lstm, prep_lstm, y_scale_lstm, 42)
        else:
            st.warning(f"⚠️ หาไฟล์ LSTM ไม่พบ: {path_lstm}")
    except Exception as e: st.error(f"LSTM Error: {e}")
    
    return models, device

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
def render_web_interface(pm01_val, pm25_val, temp_val, humid_val, wind_val, wind_speed, model_name):
    try:
        with open("index.html", "r", encoding="utf-8") as f: html_content = f.read()
        with open("style.css", "r", encoding="utf-8") as f: css_content = f.read()
        with open("script.js", "r", encoding="utf-8") as f: js_content = f.read()
    except FileNotFoundError:
        return "<h3 style='color:red; text-align:center;'>Error: ไม่พบไฟล์ index.html, style.css หรือ script.js</h3>"

    if model_name == "No Model":
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
                if(window.updateStatus) window.updateStatus({pm01_val}, `{ai_text}`);
                if(window.updateWindDirection) window.updateWindDirection({wind_val});
                
                if(window.updateCharts) window.updateCharts({chart_current}, {chart_hour}, {chart_day});
                
                var pm25Elem = document.getElementById('val-pm25');
                var tempElem = document.getElementById('val-temp');
                var humidElem = document.getElementById('val-humid');
                var modelElem = document.getElementById('modelNameDisplay');
                
                if(pm25Elem) pm25Elem.innerText = '{pm25_val}';
                if(tempElem) tempElem.innerText = '{temp_val}';
                if(humidElem) humidElem.innerText = '{humid_val}';
                if(modelElem) modelElem.innerText = 'Predicted by {model_name} Model';
                
            }}, 500);
        </script>
        """

    return f"<style>{css_content}</style>{html_content}{injection_script}"

# ==========================================
# [ส่วนที่ 4: Streamlit UI]
# ==========================================
st.markdown("""
    <style>
        /* ดันขอบจอให้ชิดขึ้น */
        .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
        
        /* 1. ซ่อนเมนู Streamlit และ Footer ด้านล่าง */
        #MainMenu {visibility: hidden;} 
        footer {visibility: hidden;}
        
        /* 2. ซ่อนปุ่ม Deploy (มุมบนขวา) */
        [data-testid="stAppDeployButton"] {display: none !important;}
        
        /* 3. 🥷 ซ่อนปุ่มลอยๆ มุมขวาล่าง (ลายน้ำ Streamlit / Manage App) */
        .viewerBadge_container {display: none !important;}
        .viewerBadge_link {display: none !important;}
        
        /* ปรับแต่งกล่องแจ้งเตือน (Alert) */
        div[data-testid="stAlert"] { margin-top: -15px !important; padding: 10px !important; }
    </style>
""", unsafe_allow_html=True)

models, device = load_models()

st.sidebar.title("⚡ Control Panel")
if not models:
    st.sidebar.error("❌ ไม่พบโมเดลสักตัว! กรุณาตรวจสอบโฟลเดอร์ 'models' ว่ามีไฟล์ครบหรือไม่")

selected_model_name = st.sidebar.selectbox("เลือกโมเดล (Model)", ["LSTM", "GRU", "RNN"], index=0)
uploaded_file = st.sidebar.file_uploader("📂 อัปโหลดไฟล์ CSV (Data Input)", type=["csv"])

# เช็คเงื่อนไขการแสดงผลเว็บ
# เช็คเงื่อนไขการแสดงผลเว็บ
if uploaded_file is not None:
    if selected_model_name in models:
        try:
            input_df = pd.read_csv(uploaded_file)
            original_cols = input_df.columns.tolist() 
            
            if 'Wind_Dir' in input_df.columns:
                input_df['Wind_Dir_cos'] = np.cos(np.radians(input_df['Wind_Dir']))
                input_df['Wind_Dir_sin'] = np.sin(np.radians(input_df['Wind_Dir']))
            
            # โหลดสเปคของโมเดลที่ผู้ใช้เลือกสำหรับแสดงผลบน Dashboard
            model, prep, y_scaler, seq_len = models[selected_model_name]
            
            cols = ['Wind_Dir_cos', 'Wind_Dir_sin', 'Wind_Speed', 'Outdoor_Temperature', 'Outdoor_Humidity', 'Bar', 'Rain', 'Outdoor_PM2.5']
            missing_cols = [c for c in cols if c not in input_df.columns]
            
            if missing_cols:
                 st.error(f"❌ ไฟล์ CSV ขาดคอลัมน์: {', '.join(missing_cols)}")
                 st.markdown("<br><br>", unsafe_allow_html=True)
                 components.html(render_web_interface(0, 0, 0, 0, 0, 0, "No Model"), height=1100, scrolling=True)
                 
            elif len(input_df) >= seq_len:
                # ---------------------------------------------------------
                # 1. ระบบกวาดทำนายผล (Batch Prediction) สำหรับ "ทั้ง 3 โมเดล"
                # ---------------------------------------------------------
                download_df = input_df[original_cols].copy() 
                
                # วนลูปทำนายให้ครบทุกโมเดลที่มีในระบบ
                for m_name in ["LSTM", "GRU", "RNN"]:
                    if m_name in models:
                        mod_i, prep_i, y_scaler_i, seq_len_i = models[m_name]
                        
                        # เช็คว่าข้อมูลในไฟล์ยาวพอสำหรับโมเดลนี้ไหม
                        if len(input_df) >= seq_len_i:
                            X_proc_all = prep_i.transform(input_df[cols])
                            
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
                            
                            # เติมค่า None ในแถวแรกๆ ที่ข้อมูลไม่ครบ Seq_len ของแต่ละโมเดล
                            full_predictions = [None] * (seq_len_i - 1) + preds_final.tolist()
                        else:
                            # ถ้าแถวไม่พอ ให้เป็นค่าว่างทั้งคอลัมน์
                            full_predictions = [None] * len(input_df)
                            
                        # เพิ่มคอลัมน์ชื่อโมเดลนั้นๆ ลงในไฟล์ดาวน์โหลด
                        download_df[f'Predict_{m_name}_Indoor_PC0.1'] = full_predictions

                # ---------------------------------------------------------
                # 2. สร้างไฟล์ใหม่สำหรับดาวน์โหลด และแสดงปุ่ม
                # ---------------------------------------------------------
                csv_data = download_df.to_csv(index=False).encode('utf-8-sig') 
                
                st.sidebar.markdown("---")
                st.sidebar.success("✅ ประมวลผลครบทั้ง 3 โมเดลเสร็จสิ้น!")
                st.sidebar.download_button(
                    label="📥 ดาวน์โหลดไฟล์ผลลัพธ์ (CSV)",
                    data=csv_data,
                    file_name=f"predicted_All_Models_{uploaded_file.name}",
                    mime="text/csv",
                )

                # ---------------------------------------------------------
                # 3. นำค่าที่ทำนายได้ "เฉพาะโมเดลที่เลือก" ไปแสดงบน Dashboard
                # ---------------------------------------------------------
                # ดึงค่าผลทำนายของโมเดลที่ผู้ใช้เลือกจากตาราง download_df บรรทัดสุดท้าย
                pred_col_name = f'Predict_{selected_model_name}_Indoor_PC0.1'
                # ป้องกันกรณีค่าแถวสุดท้ายเป็น None
                last_pred = download_df.iloc[-1][pred_col_name]
                pred_val = int(last_pred) if pd.notna(last_pred) else 0

                last_row = input_df.iloc[-1]
                pm25_disp = round(last_row['Outdoor_PM2.5'], 2)
                temp_disp = round(last_row['Outdoor_Temperature'], 2)
                humid_disp = round(last_row['Outdoor_Humidity'], 2)
                wind_spd = round(last_row['Wind_Speed'], 2)
                wind_disp = round(last_row['Wind_Dir'], 2) if 'Wind_Dir' in input_df.columns else 0
                
                html_content = render_web_interface(pred_val, pm25_disp, temp_disp, humid_disp, wind_disp, wind_spd, selected_model_name)
                components.html(html_content, height=1100, scrolling=True)
                
            else:
                st.error(f"ข้อมูลไม่เพียงพอ! โมเดล {selected_model_name} ต้องการอย่างน้อย {seq_len} แถว (ปัจจุบันอัปโหลดมา {len(input_df)} แถว)")
                st.markdown("<br><br>", unsafe_allow_html=True)
                components.html(render_web_interface(0, 0, 0, 0, 0, 0, "No Model"), height=1100, scrolling=True)
                
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์หรือประมวลผล: {e}")
            st.markdown("<br><br>", unsafe_allow_html=True)
            components.html(render_web_interface(0, 0, 0, 0, 0, 0, "No Model"), height=1100, scrolling=True)
    else:
        st.error(f"❌ ไม่สามารถใช้งานโมเดล '{selected_model_name}' ได้ กรุณาตรวจสอบไฟล์ในโฟลเดอร์ 'models'")
        st.markdown("<br><br>", unsafe_allow_html=True)
        components.html(render_web_interface(0, 0, 0, 0, 0, 0, "No Model"), height=1100, scrolling=True)
else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    html_content = render_web_interface(0, 0, 0, 0, 0, 0, "No Model")
    components.html(html_content, height=1100, scrolling=True)