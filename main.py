import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import time

# --- IMPORT DATABASE ---
from jajanan_data import CLASS_NAMES, JAJANAN_DB

# --- KONFIGURASI ---
st.set_page_config(page_title="Pendeteksi Jajanan Jatim", page_icon="🍰", layout="centered")

MODEL_PATH = 'model_jajanan.h5'
TARGET_SIZE = (224, 224)

# --- 1. LOAD MODEL (Ditambahkan compile=False untuk cegah crash versi TF) ---
@st.cache_resource
def load_my_model():
    try:
        # compile=False terbukti ampuh mengatasi bug versi Keras/TF di server deploy
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"⚠️ Gagal memuat file model '{MODEL_PATH}' di server.")
        st.exception(e) # Menampilkan detail error asli tanpa di-redact Streamlit
        return None

loaded_model = load_my_model()

# --- 2. LOGIKA PREDIKSI ---
def run_prediction(img, model_tf):
    image_resized = ImageOps.fit(img, TARGET_SIZE, Image.Resampling.LANCZOS)
    img_array = np.asarray(image_resized).astype('float32') / 255.0
    img_expand = np.expand_dims(img_array, axis=0)
    
    predictions = model_tf.predict(img_expand)
    idx = np.argmax(predictions[0])
    conf = np.max(predictions[0]) * 100
    
    if idx >= len(CLASS_NAMES):
        return f"Unknown_{idx}", conf
    return CLASS_NAMES[idx], conf

# --- NAVIGASI ---
if 'stage' not in st.session_state:
    st.session_state['stage'] = 'landing'
if 'result_data' not in st.session_state:
    st.session_state['result_data'] = None
if 'active_img' not in st.session_state:
    st.session_state['active_img'] = None

def go_to(stage_name):
    st.session_state['stage'] = stage_name
    st.rerun()

# ==========================================
# --- STAGE 1: LANDING PAGE ---
# ==========================================
if st.session_state['stage'] == 'landing':
    st.write("##")
    st.markdown("<h1 style='text-align: center;'>🍰<br>Sistem Deteksi Jajanan</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Kenali kekayaan kuliner Jawa Timur dengan teknologi AI.</p>", unsafe_allow_html=True)
    st.write("---")
    if st.button("Mulai Deteksi Sekarang", use_container_width=True, type="primary"):
        go_to('detection')

# ==========================================
# --- STAGE 2: DETECTION PAGE ---
# ==========================================
elif st.session_state['stage'] == 'detection':
    st.title("Pindai Jajanan 📸")
    
    # Kamera Otomatis
    cam_file = st.camera_input("Ambil foto jajanan")
    
    st.markdown("<h5 style='text-align: center; color: #888; margin: 20px 0;'>— Atau —</h5>", unsafe_allow_html=True)
    
    # Galeri Otomatis
    gal_file = st.file_uploader("Upload dari galeri", type=["jpg", "png", "jpeg"])

    final_img = None
    if cam_file: final_img = Image.open(cam_file)
    elif gal_file: final_img = Image.open(gal_file)

    if final_img:
        st.session_state['active_img'] = final_img
        st.write("---")
        if st.button("🔥 Analisis Gambar", use_container_width=True, type="primary"):
            # SAFEGUARD: Pastikan model beneran ada sebelum running prediksi
            if loaded_model is not None:
                res, cf = run_prediction(final_img, loaded_model)
                st.session_state['result_data'] = {'name': res, 'conf': cf}
                go_to('results')
            else:
                st.error("❌ Fitur analisis tidak dapat dijalankan karena model .h5 gagal dimuat sempurna. Silakan cek pesan error detail di bagian atas aplikasi.")
            
    if st.button("Kembali ke Home", use_container_width=True):
        go_to('landing')

# ==========================================
# --- STAGE 3: RESULTS PAGE ---
# ==========================================
elif st.session_state['stage'] == 'results':
    data = st.session_state['result_data']
    img = st.session_state['active_img']
    
    st.title("Hasil Pemindaian ✨")
    
    st.image(img, use_container_width=True)
    
    if data:
        key = data['name']
        if key in JAJANAN_DB:
            info = JAJANAN_DB[key]
            st.success(f"### {info['nama_display']}")
            
            col1, col2 = st.columns(2)
            col1.metric("Confidence", f"{data['conf']:.1f}%")
            col2.metric("Kota Asal", info['asal'])
            
            with st.expander("Informasi Lengkap", expanded=True):
                st.write(f"**Deskripsi:** {info['deskripsi']}")
                st.write(f"**Bahan Utama:** {info['bahan']}")
        else:
            st.warning(f"Terdeteksi sebagai: {key.title()}")
            st.metric("Confidence Score", f"{data['conf']:.1f}%")
            
    st.write("---")
    
    if st.button("🔄 Scan Ulang Jajanan", use_container_width=True, type="primary"):
        st.session_state['active_img'] = None
        go_to('detection')
        
    if st.button("🏠 Kembali ke Home", use_container_width=True):
        go_to('landing')