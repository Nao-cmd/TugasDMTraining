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
        model = pickle.load(open('ensemble_model_hoax.pkl', 'rb')) # Model Soft Voting lama
        tfidf = pickle.load(open('tfidf_vectorizer_hoax.pkl', 'rb'))
        return model, tfidf
    except FileNotFoundError:
        st.error("🚨 Error: File model (.pkl) tidak ditemukan.")
        st.stop()
        
ensemble_model, tfidf = load_artifacts()

# Ambil nilai akurasi final
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

# ... (Input Form dihilangkan untuk brevity)

if st.button('🎯 Prediksi Status Berita'):
    if not input_text:
        st.warning("Mohon masukkan teks berita terlebih dahulu.")
    else:
        with st.spinner('Memproses dan memprediksi...'):
            processed_text = preprocess_text(input_text)
            input_vector = tfidf.transform([processed_text]) 
            
            # ----------------------------------------------------
            # 8. PERBAIKAN BUG PREDICT_PROBA: MANUAL HARD VOTING
            # ----------------------------------------------------
            
            # Kumpulkan prediksi biner dari setiap model individu (RF & SVM)
            predictions_list = []
            for estimator in ensemble_model.estimators_:
                predictions_list.append(estimator.predict(input_vector))
            
            # Ubah ke array NumPy (bentuk 2x1)
            all_predictions = np.array(predictions_list).T 
            
            # Hitung suara terbanyak (Hard Voting)
            # np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), 1, all_predictions)
            final_vote = np.argmax(np.bincount(all_predictions[0]))
            
            prediction = final_vote # Hasil: 0 atau 1
            
            # 9. Result
            st.subheader("Hasil Klasifikasi")
            
            if prediction == 1:
                st.error('❌ Status Prediksi: HOAX')
                st.write("Sistem memprediksi berita ini **TIDAK BENAR** (Hoax).")
            else:
                st.success('✅ Status Prediksi: FAKTA (Valid)')
                st.write("Sistem memprediksi berita ini **BENAR** (Valid).")
