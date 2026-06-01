import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import time
import os

# --- IMPORT DATABASE ---
from jajanan_data import CLASS_NAMES, JAJANAN_DB

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Pendeteksi Jajanan Jatim", page_icon="🍰", layout="centered")

MODEL_PATH = 'model_jajanan.h5'
TARGET_SIZE = (224, 224)

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_my_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        return None

loaded_model = load_my_model()

# --- 2. LOGIKA PREDIKSI (SUDAH DIPERBAIKI SINKRON DENGAN COLAB) ---
def run_prediction(img, model_tf):
    # 1. Pastikan gambar dikonversi ke RGB (antisipasi format PNG/RGBA)
    img_rgb = img.convert('RGB')
    
    # 2. Gunakan .resize langsung dengan BILINEAR agar sama dengan Keras load_img (squish format)
    image_resized = img_rgb.resize(TARGET_SIZE, Image.Resampling.NEAREST)
    
    # 3. Konversi ke array dan normalisasi skala 1/255
    img_array = np.asarray(image_resized).astype('float32') / 255.0
    
    # 4. Tambah dimensi batch (1, 224, 224, 3)
    img_expand = np.expand_dims(img_array, axis=0)
    
    # 5. Eksekusi prediksi
    predictions = model_tf.predict(img_expand)
    idx = np.argmax(predictions[0])
    conf = np.max(predictions[0]) * 100
    
    if idx >= len(CLASS_NAMES):
        return f"Unknown_{idx}", conf
    return CLASS_NAMES[idx], conf

# --- LOGIKA DIAGNOSTIK DI SIDEBAR ---
with st.sidebar:
    st.header("🔍 Status Server AI")
    if os.path.exists(MODEL_PATH):
        file_size_kb = os.path.getsize(MODEL_PATH) / 1024
        st.success(f"Model File: **Aman ({file_size_kb:.1f} KB)**")
        if loaded_model is not None:
            st.success("Status Model: **Siap Digunakan ✅**")
        else:
            st.error("Status Model: **Gagal Load ❌**")
    else:
        st.error("Model File: **Tidak Ditemukan ❌**")

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
    st.write("Silakan izinkan kamera atau gunakan galeri di bawah jika kamera bermasalah.")
    st.write("##")
    
    # 1. Kamera Otomatis Bawaan Streamlit
    cam_file = st.camera_input("Ambil foto jajanan langsung")
    
    if not cam_file:
        st.info("💡 **Kamera tidak aktif?** Jika akses kamera ditolak atau tidak tersedia, kamu bisa langsung pakai tombol **Upload dari Galeri** di bawah ini 👇")

    st.markdown("<h5 style='text-align: center; color: #888; margin: 25px 0;'>— ATAU PILIH FILE —</h5>", unsafe_allow_html=True)
    
    # 2. File Uploader Asli/Original Streamlit
    gal_file = st.file_uploader("Upload gambar dari galeri / file manager", type=["jpg", "png", "jpeg"])

    final_img = None
    if cam_file: 
        final_img = Image.open(cam_file)
    elif gal_file: 
        gal_file.seek(0) # Reset pointer file buffer
        final_img = Image.open(gal_file)

    if final_img:
        st.session_state['active_img'] = final_img
        st.write("---")
        if st.button("🔥 Analisis Gambar", use_container_width=True, type="primary"):
            if loaded_model is not None:
                res, cf = run_prediction(final_img, loaded_model)
                st.session_state['result_data'] = {'name': res, 'conf': cf}
                go_to('results')
            else:
                st.error("❌ Model gagal dimuat sempurna. Periksa status di sidebar.")
            
    if st.button("Kembali ke Home", use_container_width=True):
        go_to('landing')

# ==========================================
# --- STAGE 3: RESULTS PAGE ---
# ==========================================
elif st.session_state['stage'] == 'results':
    data = st.session_state['result_data']
    img = st.session_state['active_img']
    
    st.title("Hasil Pemindaian ✨")
    
    # Tampilan Gambar Gede Menghabiskan Layar HP
    st.image(img, use_container_width=True)
    
    if data:
        key = data['name']
        persen = data['conf']
        
        # --- LOGIKA THRESHOLD CONFIDENCE < 50% ---
        if persen < 50.0:
            st.error("### ⚠️ Jajan tidak ditemukan")
            st.warning("Harap tunggu sampai update selanjutnya.")
            st.info("💡 **Tips:** Coba ambil foto ulang dengan posisi lebih dekat, objek fokus di tengah, dan pastikan pencahayaan terang.")
            st.metric("Confidence Score (Terlalu Rendah)", f"{persen:.1f}%")
        else:
            # Jika di atas atau sama dengan 50%, info jajanan baru dimunculkan
            if key in JAJANAN_DB:
                info = JAJANAN_DB[key]
                st.success(f"### {info['nama_display']}")
                
                col1, col2 = st.columns(2)
                col1.metric("Confidence", f"{persen:.1f}%")
                col2.metric("Kota Asal", info['asal'])
                
                with st.expander("Informasi Lengkap", expanded=True):
                    st.write(f"**Deskripsi:** {info['deskripsi']}")
                    st.write(f"**Bahan Utama:** {info['bahan']}")
            else:
                st.warning(f"Terdeteksi sebagai: {key.title()}")
                st.metric("Confidence Score", f"{persen:.1f}%")
            
    st.write("---")
    
    if st.button("🔄 Scan Ulang Jajanan", use_container_width=True, type="primary"):
        st.session_state['active_img'] = None
        st.session_state['result_data'] = None
        go_to('detection')
        
    if st.button("🏠 Kembali ke Home", use_container_width=True):
        st.session_state['active_img'] = None
        st.session_state['result_data'] = None
        go_to('landing')