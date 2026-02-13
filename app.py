import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. Load Model and Scaler ---
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please ensure 'best_model.pkl' and 'scaler.pkl' are in the same directory.")
        st.stop()

model, scaler = load_model_and_scaler()

# --- 2. Define Mappings and Feature Order ---
pendidikan_map = {'D3': 0, 'S1': 1, 'SMA': 2, 'SMK': 3}
jurusan_map = {'Administrasi': 0, 'Desain Grafis': 1, 'Otomotif': 2, 'Teknik Las': 3, 'Teknik Listrik': 4}

# This order must match X_train's column order exactly
feature_cols_order = [
    'Usia',
    'Durasi_Jam',
    'Nilai_Ujian',
    'Pendidikan',
    'Jurusan',
    'Jenis_Kelamin_Laki-laki',
    'Jenis_Kelamin_Wanita',
    'Status_Bekerja_Belum Bekerja',
    'Status_Bekerja_Sudah Bekerja'
]

# --- 3. Streamlit App Interface ---
st.title('Prediksi Gaji Pertama Lulusan Pelatihan Vokasi')
st.write('Masukkan data peserta untuk memprediksi perkiraan gaji pertama mereka (dalam Juta IDR).')

# Input fields
usia = st.slider('Usia', 18, 60, 25) # Assuming age range 18-60 based on typical training data
durasi_jam = st.slider('Durasi Pelatihan (Jam)', 20, 100, 60) # Assuming duration range 20-100
nilai_ujian = st.slider('Nilai Ujian', 50.0, 100.0, 75.0)
pendidikan = st.selectbox('Pendidikan Terakhir', list(pendidikan_map.keys()))
jurusan = st.selectbox('Jurusan Pelatihan', list(jurusan_map.keys()))
jenis_kelamin = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Wanita'])
status_bekerja = st.selectbox('Status Bekerja', ['Belum Bekerja', 'Sudah Bekerja'])

# Create a dictionary for new input
new_input_raw = {
    'Usia': usia,
    'Durasi_Jam': durasi_jam,
    'Nilai_Ujian': nilai_ujian,
    'Pendidikan': pendidikan,
    'Jurusan': jurusan,
    'Jenis_Kelamin': jenis_kelamin,
    'Status_Bekerja': status_bekerja
}

# Convert to DataFrame
new_df = pd.DataFrame([new_input_raw])

# --- 4. Preprocessing Input Data ---

# Apply Label Encoding for Pendidikan and Jurusan
new_df['Pendidikan'] = new_df['Pendidikan'].map(pendidikan_map)
new_df['Jurusan'] = new_df['Jurusan'].map(jurusan_map)

# Apply One-Hot Encoding for Jenis_Kelamin and Status_Bekerja
one_hot_cols_expected = [
    'Jenis_Kelamin_Laki-laki',
    'Jenis_Kelamin_Wanita',
    'Status_Bekerja_Belum Bekerja',
    'Status_Bekerja_Sudah Bekerja'
]

for col in one_hot_cols_expected:
    new_df[col] = 0 # Initialize all with 0

if new_df['Jenis_Kelamin'].iloc[0] == 'Laki-laki':
    new_df['Jenis_Kelamin_Laki-laki'] = 1
elif new_df['Jenis_Kelamin'].iloc[0] == 'Wanita':
    new_df['Jenis_Kelamin_Wanita'] = 1

if new_df['Status_Bekerja'].iloc[0] == 'Belum Bekerja':
    new_df['Status_Bekerja_Belum Bekerja'] = 1
elif new_df['Status_Bekerja'].iloc[0] == 'Sudah Bekerja':
    new_df['Status_Bekerja_Sudah Bekerja'] = 1

# Drop original categorical columns after one-hot encoding
new_df = new_df.drop(columns=['Jenis_Kelamin', 'Status_Bekerja'])

# Ensure column order matches training data
try:
    new_df_processed = new_df[feature_cols_order]
except KeyError as e:
    st.error(f"Error in column matching during preprocessing: {e}. Please check the input features.")
    st.stop()

# Scale the processed data
new_df_scaled = scaler.transform(new_df_processed)

# --- 5. Make Prediction ---
if st.button('Prediksi Gaji'):
    prediction = model.predict(new_df_scaled)
    st.success(f'Prediksi Gaji Pertama: **Rp {prediction[0]:.2f} Juta**')
