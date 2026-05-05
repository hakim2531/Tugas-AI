import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klasifikasi Bunga Iris", layout="wide")

# --- LOAD DATASET ---
@st.cache_data
def load_data():
    # Membaca data dari file Iris_3.csv
    try:
        df = pd.read_csv("Iris_3.csv")
        return df
    except FileNotFoundError:
        return None

# --- TRAINING MODEL ---
def train_model(df):
    # Memisahkan Fitur (X) dan Target (y)
    # Menghapus kolom 'Id' jika ada, dan kolom terakhir adalah 'Species'
    X = df.drop(columns=['Id', 'Species'], errors='ignore')
    y = df['Species']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Menggunakan KNN untuk klasifikasi
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X

# --- MAIN APP ---
def main():
    st.title("🌸 Aplikasi Klasifikasi Bunga Iris")
    st.write("Aplikasi ini memprediksi spesies bunga Iris berdasarkan fitur morfologinya menggunakan data `Iris_3.csv`.")

    df_iris = load_data()
    
    if df_iris is not None:
        model, accuracy, X_features = train_model(df_iris)

        # --- SIDEBAR UNTUK INPUT PENGGUNA ---
        st.sidebar.header("📝 Input Spesifikasi Bunga")
        
        # Slider disesuaikan dengan range nilai asli di dataset
        sl = st.sidebar.slider("Sepal Length (cm)", float(df_iris.iloc[:,1].min()), float(df_iris.iloc[:,1].max()), 5.1)
        sw = st.sidebar.slider("Sepal Width (cm)", float(df_iris.iloc[:,2].min()), float(df_iris.iloc[:,2].max()), 3.5)
        pl = st.sidebar.slider("Petal Length (cm)", float(df_iris.iloc[:,3].min()), float(df_iris.iloc[:,3].max()), 1.4)
        pw = st.sidebar.slider("Petal Width (cm)", float(df_iris.iloc[:,4].min()), float(df_iris.iloc[:,4].max()), 0.2)

        # Layout Utama: 2 Kolom
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🔮 Hasil Prediksi")
            input_data = [[sl, sw, pl, pw]]
            prediksi = model.predict(input_data)[0]
            
            # Menampilkan hasil prediksi dengan warna menarik
            st.success(f"Spesies Terdeteksi: **{prediksi}**")
            
            st.write("---")
            st.subheader("📊 Evaluasi Model")
            st.metric(label="Akurasi Model", value=f"{accuracy * 100:.2f}%")
            st.write("Model dilatih menggunakan algoritma **K-Nearest Neighbors**.")

        with col2:
            st.subheader("📈 Visualisasi Distribusi Data")
            fig, ax = plt.subplots(figsize=(7, 5))
            # Visualisasi Sepal Length vs Petal Length
            sns.scatterplot(data=df_iris, x=df_iris.columns[1], y=df_iris.columns[3], 
                            hue='Species', palette='viridis', ax=ax)
            
            # Menandai posisi input pengguna di grafik
            ax.scatter(sl, pl, color='red', marker='X', s=200, label='Posisi Input Anda')
            ax.set_title("Analisis Posisi Input pada Dataset")
            ax.legend()
            st.pyplot(fig)

        # Bagian Dataset
        if st.checkbox("Tampilkan Dataset Referensi (Iris_3.csv)"):
            st.dataframe(df_iris, use_container_width=True)
    else:
        st.error("File `Iris_3.csv` tidak ditemukan. Pastikan file berada di folder yang sama dengan script.")

if __name__ == "__main__":
    main()
