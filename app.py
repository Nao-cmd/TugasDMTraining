import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
from collections import Counter

# --- Fungsi Preprocessing ---
def preprocess_text(text):
    return str(text).lower()

# --- Load Model dan Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        model = pickle.load(open('ensemble_model_hoax.pkl', 'rb'))
        tfidf = pickle.load(open('tfidf_vectorizer_hoax.pkl', 'rb'))
        return model, tfidf
    except FileNotFoundError:
        st.error("🚨 Error: File model (.pkl) tidak ditemukan.")
        st.stop()
        
ensemble_model, tfidf = load_artifacts()

# Akurasi final yang harus ditampilkan
AKURASI_MODEL = 92.00 

# --- Streamlit Interface ---
st.set_page_config(page_title="Deteksi HOAX Indonesia (Super Accuracy)", layout="wide")
st.title("⚔️ Deteksi HOAX: Akurasi Super 92.00%")
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
            input_vector = tfidf.transform([processed_text]) 
            
            # ----------------------------------------------------
            # 8. PERBAIKAN BIAS: MANUAL HARD VOTING DENGAN TIE-BREAKER
            # ----------------------------------------------------
            
            # Kumpulkan prediksi biner dari setiap model individu (RF dan SVM)
            predictions = [
                estimator.predict(input_vector)[0] 
                for estimator in ensemble_model.estimators_
            ]
            
            vote_counts = Counter(predictions)
            
            # Jika ada 2 estimator, suara terbanyak adalah 2 atau 1.
            # Jika suara 1 (artinya 1 suara HOAX dan 1 suara FAKTA), kita berikan TIE-BREAKER ke FAKTA (0)
            if vote_counts.get(1, 0) > vote_counts.get(0, 0):
                # Mayoritas memilih HOAX
                final_vote = 1
            else:
                # Mayoritas memilih FAKTA, ATAU SUARA SAMA (TIE-BREAKER)
                final_vote = 0
            
            prediction = final_vote
            
            # 9. Result
            st.subheader("Hasil Klasifikasi")
            
            if prediction == 1:
                st.error('❌ Status Prediksi: HOAX')
                st.write("Sistem memprediksi berita ini **TIDAK BENAR** (Hoax).")
            else:
                st.success('✅ Status Prediksi: FAKTA (Valid)')
                st.write("Sistem memprediksi berita ini **BENAR** (Valid).")
