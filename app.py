import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import os
from scipy.spatial.distance import euclidean

# Tizim sozlamalari
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# --- 1. SAHIFA DIZAYNI (O‘RNATILGAN TIZIMLAR USLUBIDA) ---
st.set_page_config(page_title="O‘rnatilgan tizimlar: Biometrik Boshqaruv", page_icon="⚙️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;500;700&display=swap');

    .stApp {
        background: #050505;
        color: #00ff41; /* Terminal yashil rangi */
        font-family: 'JetBrains Mono', monospace;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        color: #00ff41;
        text-shadow: 0 0 15px rgba(0, 255, 65, 0.4);
        border-bottom: 1px solid #00ff41;
        padding-bottom: 15px;
        margin-bottom: 25px;
    }

    .cyber-panel {
        background: rgba(0, 40, 0, 0.2);
        border: 1px solid #00ff41;
        border-radius: 10px;
        padding: 20px;
        box-shadow: inset 0 0 10px rgba(0, 255, 65, 0.1);
    }

    .stButton>button {
        width: 100%;
        background: #004d00;
        color: #00ff41;
        border: 1px solid #00ff41;
        font-family: 'JetBrains Mono', monospace;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: #00ff41;
        color: #000;
        box-shadow: 0 0 20px #00ff41;
    }

    .status-monitor {
        padding: 15px;
        background: rgba(0, 255, 65, 0.05);
        border: 1px dashed #00ff41;
        border-radius: 5px;
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'step' not in st.session_state: st.session_state.step = "SETUP"
if 'master_pattern' not in st.session_state: st.session_state.master_pattern = []

# --- 3. VIDEO PROCESSING ENGINES ---

class BaseProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        # Aniqlikni oshirish uchun confidence 0.6
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.temp_points = []
    
    def get_frame(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=10) # Tiniqlashtirish
        return cv2.flip(img, 1)

class SetupProcessor(BaseProcessor):
    def recv(self, frame):
        img = self.get_frame(frame)
        h, w, _ = img.shape
        res = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if res.multi_hand_landmarks:
            for hand_lms in res.multi_hand_landmarks:
                l = hand_lms.landmark
                # Chizish buyrug'i: Ko'rsatkich barmoq yuqorida, o'rta barmoq pastda
                if l[8].y < l[6].y and l[12].y > l[10].y:
                    self.temp_points.append((l[8].x, l[8].y))
                    cv2.circle(img, (int(l[8].x*w), int(l[8].y*h)), 10, (0, 255, 65), -1)
                
                # Chiziqlarni ekranda ko'rsatish
                for i in range(1, len(self.temp_points)):
                    cv2.line(img, (int(self.temp_points[i-1][0]*w), int(self.temp_points[i-1][1]*h)),
                             (int(self.temp_points[i][0]*w), int(self.temp_points[i][1]*h)), (0, 255, 65), 3)
                
                # Skeletni chizish
                self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        
        return frame.from_ndarray(img, format="bgr24")

class ControlProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.sw, self.sh = pyautogui.size()
        self.prev_x, self.prev_y = self.sw//2, self.sh//2
        self.last_act = 0
        self.smooth_factor = 0.2

    def recv(self, frame):
        img = self.get_frame(frame)
        res = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if res.multi_hand_landmarks:
            for hand_lms in res.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
            
            hands_count = len(res.multi_hand_landmarks)
            # Desktop (Ikkala qo'l)
            if hands_count == 2:
                all_open = True
                for h_lms in res.multi_hand_landmarks:
                    if h_lms.landmark[8].y > h_lms.landmark[6].y: all_open = False
                if all_open and time.time() - self.last_act > 1.5:
                    pyautogui.hotkey('win', 'd'); self.last_act = time.time()
            
            l = res.multi_hand_landmarks[0].landmark
            # Kursor
            tx = np.interp(l[8].x, [0.3, 0.7], [0, self.sw])
            ty = np.interp(l[8].y, [0.3, 0.7], [0, self.sh])
            cx = self.prev_x + (tx - self.prev_x) * self.smooth_factor
            cy = self.prev_y + (ty - self.prev_y) * self.smooth_factor
            
            if l[8].y < l[6].y:
                pyautogui.moveTo(cx, cy)
                self.prev_x, self.prev_y = cx, cy

            now = time.time()
            d48 = np.hypot(l[4].x - l[8].x, l[4].y - l[8].y)     
            d412 = np.hypot(l[4].x - l[12].x, l[4].y - l[12].y)  
            d416 = np.hypot(l[4].x - l[16].x, l[4].y - l[16].y)  
            d420 = np.hypot(l[4].x - l[20].x, l[4].y - l[20].y)  

            if d48 < 0.04 and now - self.last_act > 0.5:
                pyautogui.click(); self.last_act = now
            elif d412 < 0.04 and now - self.last_act > 0.8:
                pyautogui.hotkey('win', 'tab'); self.last_act = now
            elif d416 < 0.04 and now - self.last_act > 0.7:
                pyautogui.rightClick(); self.last_act = now
            elif d420 < 0.04 and now - self.last_act > 1.5:
                os.system("start winword"); self.last_act = now
            
            # Scroll
            if l[8].y > l[6].y and l[12].y > l[10].y and l[16].y > l[14].y and l[20].y > l[18].y:
                if l[0].y < 0.45: pyautogui.scroll(150)
                elif l[0].y > 0.55: pyautogui.scroll(-150)

        return frame.from_ndarray(img, format="bgr24")

# --- 4. ASOSIY INTERFEYS ---
st.markdown("<h1 class='main-title'>O‘RNATILGAN TIZIMLAR: BIOMETRIK NAZORAT</h1>", unsafe_allow_html=True)

RTC_CONF = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
MEDIA_CONSTRAINTS = {"video": {"width": 1280, "height": 720}, "audio": False}

col1, col2 = st.columns([7, 3])

with col2:
    st.markdown("<div class='cyber-panel'>", unsafe_allow_html=True)
    st.write("### 🎛️ BOSHQARUV PANELI")
    
    if st.session_state.step == "SETUP":
        st.write("🟢 **Tizim:** Kalit yaratish kutilmoqda...")
        ctx = webrtc_streamer(key="setup", video_processor_factory=SetupProcessor, 
                              rtc_configuration=RTC_CONF, media_stream_constraints=MEDIA_CONSTRAINTS)
        
        if st.button("💾 KALITNI SAQLASH"):
            if ctx.video_processor and len(ctx.video_processor.temp_points) > 15:
                st.session_state.master_pattern = ctx.video_processor.temp_points.copy()
                st.session_state.step = "AUTH"
                st.rerun()
            else:
                st.error("Chiziq yetarli emas! (Kamida 15 ta nuqta)")

    elif st.session_state.step == "AUTH":
        st.write("🟡 **Tizim:** Identifikatsiya kutilmoqda...")
        ctx_a = webrtc_streamer(key="auth", video_processor_factory=SetupProcessor, 
                                rtc_configuration=RTC_CONF, media_stream_constraints=MEDIA_CONSTRAINTS)
        
        if st.button("🔓 TASDIQLASH"):
            if ctx_a.video_processor:
                m_pat = st.session_state.master_pattern
                t_pat = ctx_a.video_processor.temp_points
                
                if len(t_pat) > 15:
                    # Matematik solishtirish (Namuna olish orqali)
                    idx1 = np.linspace(0, len(m_pat)-1, 20).astype(int)
                    idx2 = np.linspace(0, len(t_pat)-1, 20).astype(int)
                    v1 = np.array([m_pat[i] for i in idx1])
                    v2 = np.array([t_pat[i] for i in idx2])
                    
                    error_dist = np.mean([euclidean(v1[i], v2[i]) for i in range(20)])
                    
                    if error_dist < 0.18: # 82% dan yuqori aniqlik
                        st.session_state.step = "CONTROL"
                        st.rerun()
                    else:
                        st.error(f"Xatolik yuqori: {round(error_dist, 2)}. Qayta chizing!")
                        ctx_a.video_processor.temp_points = []
                else:
                    st.error("Tasdiqlash uchun chizmani bajaring!")

    elif st.session_state.step == "CONTROL":
        st.write("🔵 **Tizim:** FOYDALANUVCHI: Avazbek")
        st.markdown("""
        <div class='status-monitor'>
        <b>Faol amallar:</b><br>
        - Kursor: Ko'rsatkich barmoq<br>
        - Chap Click: Bosh+Index<br>
        - MS Word: Bosh+Jimjiloq<br>
        - Scroll: Musht harakati
        </div>
        """, unsafe_allow_html=True)
        
        webrtc_streamer(key="ctrl", video_processor_factory=ControlProcessor, 
                        rtc_configuration=RTC_CONF, media_stream_constraints=MEDIA_CONSTRAINTS)
        
        if st.button("🔴 TIZIMNI TO'XTATISH"):
            st.session_state.step = "SETUP"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with col1:
    st.markdown("<div class='cyber-panel'>", unsafe_allow_html=True)
    st.write("#### 📷 VIDEO MONITORING (O'rnatilgan Tizimlar Laboratoriyasi)")
    st.write("Ushbu darcha orqali biometrik ma'lumotlar qayta ishlanadi.")
    st.markdown("</div>", unsafe_allow_html=True)