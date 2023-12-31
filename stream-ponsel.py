import pickle
import streamlit as st

ponsel_model = pickle.load(open('estimasi_harga_HP.sav', 'rb'))

st.title('Estimasi Harga HP Berdasarkan Spesifikasi')

RAM = st.number_input('Kapasitas RAM (GB)')
Cpu_Core = st.number_input('Jumlah Core CPU')
Internal = st.number_input('Total Memory Internal (GB)')
Battery = st.number_input('Kapasitas Baterai (mAh)')
FrontCam = st.number_input('Kamera Depan (Mega Pixels)')
RearCam = st.number_input('Kamera Belakang (Mega Pixels)')
Resolution = st.number_input('Ukuran Layar (INCH)')
Cpu_Freq = st.number_input('Kecepatan CPU (Ghz)')
Sale = st.number_input('Jumlah Penjualan harian (Unit)')
Weight = st.number_input('Berat Device (Grams)')
PPI = st.number_input('Tingkat Kepadatan Pixels (ppi)')
Thickness = st.number_input('Ketebalan Device (Mm)')

predict = ''

if st.button('Estimasi'):
    predict = ponsel_model.predict(
        [[RAM, Cpu_Core, Internal, Battery, FrontCam, RearCam, Resolution, Cpu_Freq, Sale, Weight, PPI, Thickness]]
    )
    st.write ('Estimasi harga dalam hitungan Juta : ', predict*1000)
