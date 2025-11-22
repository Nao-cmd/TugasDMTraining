import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np

# --- Fungsi Preprocessing ---
def preprocess_text(text):
    return str(text).lower()

# --- Load Model dan Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        # Nama file PKL baru
        model = pickle.load(open('ensemble_model_hoax.pkl', 'rb')) 
        tfidf = pickle.load(open('tfidf_vectorizer_hoax.pkl', 'rb'))
        return model, tfidf
    except FileNotFoundError:
        st.error("🚨 Error: File model (.pkl) tidak ditemukan.")
        st.stop()
        
ensemble_model, tfidf = load_artifacts()

# Akurasi final (Gunakan hasil dari output training GBC+RF)
# Harap ganti 92.00 dengan angka hasil training yang baru
AKURASI_MODEL = 91.00 

# --- Streamlit Interface ---
st.set_page_config(page_title="Deteksi HOAX Indonesia (Super Accuracy)", layout="wide")
st.title("⚔️ Deteksi HOAX: Akurasi Super (RF + GBC)")
st.subheader("Aplikasi Cerdas Klasifikasi Berita HOAX vs FAKTA")

# 4. Model Performance (Di Sidebar)
st.sidebar.header("Model Performance")
st.sidebar.metric(label="Akurasi Model di Data Tes", value=f"{AKURASI_MODEL:.2f}%")
if AKURASI_MODEL >= 90:
    st.sidebar.success("🏆 Akurasi Super Tercapai (>90%)!")
else:
    st.sidebar.warning("⚠️ Akurasi masih di bawah 90%.")

# 5. Input Data Form
st.header("Input Teks Berita Baru")
input_text = st.text_area("Masukkan Teks Berita yang Ingin Divalidasi:", 
                          "Contoh: Minyak goreng dari sawit dapat menyembuhkan semua penyakit virus yang baru-baru ini menyebar.", 
                          height=150)

if st.button('🎯 Prediksi Status Berita'):
    if not input_text:
        st.warning("Mohon masukkan teks berita terlebih dahulu.")
    else:
        with st.spinner('Memproses dan memprediksi...'):
            processed_text = preprocess_text(input_text)
            
            # PERBAIKAN KRITIS: Mengonversi input ke Dense
            input_sparse = tfidf.transform([processed_text])
            input_vector = input_sparse.toarray() 
            
            # 8. Prediksi (menggunakan voting='soft' yang sekarang stabil)
            prediction = ensemble_model.predict(input_vector)[0]
            
            # 9. Result
            st.subheader("Hasil Klasifikasi")
            
            if prediction == 1:
                st.error('❌ Status Prediksi: HOAX')
                st.write("Sistem memprediksi berita ini **TIDAK BENAR** (Hoax).")
            else:
                st.success('✅ Status Prediksi: FAKTA (Valid)')
                st.write("Sistem memprediksi berita ini **BENAR** (Valid).")
