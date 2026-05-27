import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import time

# --- KONFIGURASI HALAMAN ---
# Gua bikin layout centered biar mirip mockup HP lu
st.set_page_config(
    page_title="Pendeteksi Jajanan Tradisional",
    page_icon="🍰",
    layout="centered"
)

# --- KONFIGURASI MODEL & DATA (Sesuaikan dengan Colab lu) ---
MODEL_PATH = 'model_jajanan.h5'
TARGET_SIZE = (224, 224) # Harus sama dengan input_shape saat training
# Urutan HARUS sama persis dengan urutan class di Colab
CLASS_NAMES = ['Klepon', 'Lumpia', 'Onde-onde', 'Pastel', 'Serabi'] 

# ==========================================
# --- FUNGSI-FUNGSI UTAMA (Jangan Diubah) ---
# ==========================================

# 1. Fungsi Memuat Model (di-cache biar cepet)
@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"⚠️ File model '{MODEL_PATH}' tidak ditemukan atau rusak. Pastikan file ada di folder yang sama di GitHub. Error: {e}")
        return None

# 2. Fungsi Proses Prediksi
def predict_image(img, model_tf):
    # Preprocessing
    image_resized = ImageOps.fit(img, TARGET_SIZE, Image.Resampling.LANCZOS)
    img_array = np.asarray(image_resized)
    img_array = img_array.astype('float32')
    
    # NORMALISASI: Matikan baris bawah jika model lu udah punya layer Rescaling internal
    img_array = img_array / 255.0
    
    img_expand = np.expand_dims(img_array, axis=0)

    # Prediksi
    with st.spinner('Sedang menganalisis gambar... 🧐'):
        time.sleep(1) # Efek dramatis dikit
        predictions = model_tf.predict(img_expand)
        
    score = tf.nn.softmax(predictions[0]) # Opsional, jika output logits
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    return CLASS_NAMES[predicted_class_idx], confidence

# ==========================================
# --- LOGIKA SISTEM HALAMAN (SESSION STATE) ---
# ==========================================

# Inisialisasi status halaman jika belum ada
if 'app_stage' not in st.session_state:
    st.session_state['app_stage'] = 'landing' # Default: Halaman Depan

# Fungsi buat ganti halaman
def change_stage(stage_name):
    st.session_state['app_stage'] = stage_name
    st.rerun() # Refresh app buat nampilin stage baru

# ==========================================
# --- TAMPILAN INTERFACE ---
# ==========================================

# Mencoba load model di awal biar cepet
loaded_model = load_my_model()

# --- STAGE 1: HALAMAN DEPAN (LANDING) ---
if st.session_state['app_stage'] == 'landing':
    
    # Membuat spasi kosong di atas agar konten agak ke tengah (Vertical Centering)
    st.write("##")
    st.write("##")

    # Layouting pake kolom kosong biar konten bener-bener di tengah layaknya HP
    left, mid, right = st.columns([1, 4, 1])
    
    with mid:
        # 1. Visual Placeholder (Mirip stacked bars di mockup lu)
        # Gua pake markdown dan emoji biar ga perlu file gambar tambahan
        st.markdown(
            """
            <div style="text-align: center; font-size: 60px; line-height: 0.8; color: #555; margin-bottom: -15px;">
                🍱<br>
                ➖<br>
                ➖
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.write("---") # Garis pemisah tipis

        # 2. Teks Selamat Datang (Sesuaikan tulisan 'blablabla' lu di sini)
        st.markdown(
            """
            <div style="text-align: center;">
                <h1 style="font-size: 28px; margin-bottom: 5px;">Selamat Datang! 👋</h1>
                <p style="font-size: 16px; color: #666;">
                    Di Aplikasi Pendeteksi Jajanan Tradisional.<br>
                    Cukup upload foto, kami akan tebak jenis jajanannya.<br>
                    Siap berpetualang kuliner?
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("##") # Spasi

        # 3. Tombol Utama (Styled agar lebar dan rounded mirip mockup)
        # st.button secara default udah agak rounded di UI modern Streamlit
        if st.button("Mulai deteksi", use_container_width=True, type="secondary"):
            if loaded_model is not None:
                change_stage('detection') # Pindah stage
            else:
                st.error("Model belum siap, tidak bisa lanjut.")

# --- STAGE 2: HALAMAN DETEKSI (PITCH) ---
elif st.session_state['app_stage'] == 'detection':
    
    # Navigasi Back
    if st.button("⬅️ Kembali ke Home"):
        change_stage('landing')
        
    st.write("---")
    st.title("Pindai Jajanan Lu 📸")
    st.write("Silakan upload foto jajanan tradisional Indonesia (maks. 200MB).")

    # Fitur Upload
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Buka dan Tampilkan Gambar
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar ter-upload', use_container_width=True)
        st.write("---")

        # Tombol untuk memicu prediksi (biar ga prediksi otomatis terus)
        if st.button("Analisis Gambar", use_container_width=True, type="primary"):
            if loaded_model is not None:
                # Panggil Fungsi Prediksi
                hasil, persen = predict_image(image, loaded_model)

                # Tampilkan Hasil
                st.subheader("Hasil Analisis:")
                st.success(f"Prediksi: **{hasil}**")
                st.metric(label="Tingkat Keyakinan", value=f"{persen:.2f}%")
            else:
                st.error("Model tidak tersedia.")

    # Footer tipis
    st.write("##")
    st.write("##")
    st.markdown("<div style='text-align:center; color:#aaa; font-size:12px;'>Made with ❤️ for Indonesian Culture</div>", unsafe_allow_html=True)