import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import time

# --- IMPORT DATA DARI FILE TERPISAH ---
from jajanan_data import CLASS_NAMES, JAJANAN_DB

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Pendeteksi Jajanan Tradisional",
    page_icon="🍰",
    layout="centered"
)

MODEL_PATH = 'model_jajanan.h5'
TARGET_SIZE = (224, 224) 

# --- 1. LOAD MODEL (Cached) ---
@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"⚠️ Gagal memuat model '{MODEL_PATH}'. Error: {e}")
        return None

loaded_model = load_my_model()

# --- 2. FUNGSI PROSES PREDIKSI ---
def predict_image(img, model_tf):
    image_resized = ImageOps.fit(img, TARGET_SIZE, Image.Resampling.LANCZOS)
    img_array = np.asarray(image_resized)
    img_array = img_array.astype('float32')
    
    # Normalisasi (matikan jika model lu sudah include layer Rescaling internal)
    img_array = img_array / 255.0
    img_expand = np.expand_dims(img_array, axis=0)

    with st.spinner('Sedang memindai gambar... 🧐'):
        time.sleep(1)
        predictions = model_tf.predict(img_expand)
        
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    # SAFEGUARD PROTEKSI INDEX ERROR
    if predicted_class_idx >= len(CLASS_NAMES):
        return f"Unknown_Index_{predicted_class_idx}", confidence
    
    return CLASS_NAMES[predicted_class_idx], confidence

# --- LOGIKA NAVIGASI (SESSION STATE) ---
if 'app_stage' not in st.session_state:
    st.session_state['app_stage'] = 'landing'

def change_stage(stage_name):
    st.session_state['app_stage'] = stage_name
    st.rerun()

# ==========================================
# --- STAGE 1: HALAMAN LANDING ---
# ==========================================
if st.session_state['app_stage'] == 'landing':
    st.write("##")
    st.write("##")
    left, mid, right = st.columns([1, 4, 1])
    
    with mid:
        st.markdown(
            """
            <div style="text-align: center; font-size: 65px; line-height: 0.8; margin-bottom: -10px;">
                🍰<br>➖<br>➖
            </div>
            """, unsafe_allow_html=True
        )
        st.write("---")
        st.markdown(
            """
            <div style="text-align: center;">
                <h1 style="font-size: 26px;">Selamat Datang! 👋</h1>
                <p style="font-size: 15px; color: #666;">
                    Aplikasi cerdas pendeteksi varian jajanan tradisional khas Nusantara berbasis Deep Learning.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
        st.write("##")
        if st.button("Mulai deteksi", use_container_width=True):
            change_stage('detection')

# ==========================================
# --- STAGE 2: HALAMAN DETEKSI ---
# ==========================================
elif st.session_state['app_stage'] == 'detection':
    if st.button("⬅️ Kembali ke Home"):
        change_stage('landing')
        
    st.write("---")
    st.title("Pindai Jajanan Tradisional 📸")
    st.write("Silakan ambil foto langsung atau upload file dari galeri.")
    st.write("##")

    # 1. INPUT KAMERA LANGSUNG (Otomatis trigger prompt allow/deny di browser)
    camera_file = st.camera_input("Arahkan jajanan ke kamera")
    
    # Pembatas visual biar rapi
    st.markdown("<h4 style='text-align: center; color: #aaa; margin: 25px 0;'>— ATAU —</h4>", unsafe_allow_html=True)
    
    # 2. INPUT FILE UPLOADER (Langsung stand-by buka file manager saat diklik)
    uploaded_file = st.file_uploader("📁 Upload gambar dari file manager / galeri...", type=["jpg", "jpeg", "png"])

    # Menentukan sumber gambar yang aktif (Kamera diprioritaskan jika keduanya terisi)
    image_data = None
    if camera_file is not None:
        image_data = Image.open(camera_file)
    elif uploaded_file is not None:
        image_data = Image.open(uploaded_file)

    # Jika salah satu input dapet gambar, tombol analisis bakal muncul
    if image_data is not None:
        st.write("---")
        # Jika input dari file uploader, kita tampilin preview gambarnya di bawah
        if uploaded_file is not None and camera_file is None:
            st.image(image_data, caption='Gambar ter-upload', use_container_width=True)
        
        st.write("##")
        if st.button("🔥 Analisis Gambar", use_container_width=True, type="primary"):
            if loaded_model is not None:
                hasil, persen = predict_image(image_data, loaded_model)

                if "Unknown_Index_" in hasil:
                    st.error(f"⚠️ **Error Kecocokan Model!** Model memprediksi indeks kelas **{hasil.split('_')[-1]}**, tetapi daftar CLASS_NAMES di file 'jajanan_data.py' cuma punya {len(CLASS_NAMES)} pilihan. Silakan periksa kembali urutan folder dataset di Colab lu.")
                else:
                    st.write("##")
                    st.subheader("🎉 Hasil Klasifikasi")
                    
                    if hasil in JAJANAN_DB:
                        info = JAJANAN_DB[hasil]
                        
                        # Menampilkan Output Informasi Sesuai Request Lu
                        st.metric(label="Nama Jajanan Tradisional", value=info['nama_display'])
                        st.metric(label="Tingkat Keyakinan (Confidence Score)", value=f"{persen:.2f}%")
                        
                        st.info(f"📍 **Kota Asal (Khas Jatim):** {info['asal']}")
                        st.markdown(f"""
                        **📄 Deskripsi:** {info['deskripsi']}
                        
                        **🌾 Bahan-Bahan Utama:** {info['bahan']}
                        """)
                    else:
                        st.metric(label="Nama Jajanan Tradisional", value=hasil.title())
                        st.metric(label="Tingkat Keyakinan (Confidence Score)", value=f"{persen:.2f}%")
                        st.warning("Informasi tambahan untuk jajanan ini belum dimasukkan ke database sistem.")
            else:
                st.error("Model tidak siap digunakan.")