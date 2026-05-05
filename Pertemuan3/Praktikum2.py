# %%writefile apps.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klasifikasi Bunga Iris", layout="wide")

# --- LOAD DATASET ---
@st.cache_data
def load_data():
    # Membaca data dari file lokal Iris.csv
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    try:
        df = pd.read_csv(url)
        return df
    except:
        st.error("File Iris.csv tidak ditemukan di folder!")
        return None

# --- TRAINING MODEL ---
def train_model(df):
    # Menyesuaikan nama kolom dengan file Iris.csv kamu (ada 'Id' dan 'Species')
    X = df.drop(columns=['Id', 'Species'], errors='ignore')
    y = df['Species']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

def main():
    st.title("🌸 Aplikasi Klasifikasi Bunga Iris")
    st.write("Prediksi spesies bunga Iris berdasarkan file `Iris.csv` lokal.")

    df_iris = load_data()
    
    if df_iris is not None:
        model, accuracy = train_model(df_iris)

        # --- SIDEBAR UNTUK INPUT ---
        st.sidebar.header("📝 Input Spesifikasi Bunga")
        
        # Mengambil nilai min/max dari kolom Iris.csv kamu secara otomatis
        sl = st.sidebar.slider("Sepal Length (cm)", float(df_iris['SepalLengthCm'].min()), float(df_iris['SepalLengthCm'].max()), 5.1)
        sw = st.sidebar.slider("Sepal Width (cm)", float(df_iris['SepalWidthCm'].min()), float(df_iris['SepalWidthCm'].max()), 3.5)
        pl = st.sidebar.slider("Petal Length (cm)", float(df_iris['PetalLengthCm'].min()), float(df_iris['PetalLengthCm'].max()), 1.4)
        pw = st.sidebar.slider("Petal Width (cm)", float(df_iris['PetalWidthCm'].min()), float(df_iris['PetalWidthCm'].max()), 0.2)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🔮 Hasil Prediksi")
            if st.button("Prediksi"):
                input_data = np.array([[sl, sw, pl, pw]])
                prediksi = model.predict(input_data)[0]
                st.success(f"Spesies Terdeteksi: **{prediksi}**")
            
            st.write("---")
            st.subheader("📊 Evaluasi Model")
            st.metric(label="Akurasi Model (KNN)", value=f"{accuracy * 100:.2f}%")

        with col2:
            st.subheader("📈 Visualisasi Data")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.scatterplot(data=df_iris, x='SepalLengthCm', y='PetalLengthCm', hue='Species', ax=ax)
            ax.scatter(sl, pl, color='red', marker='X', s=200, label='Posisi Input')
            ax.legend()
            st.pyplot(fig)

        if st.checkbox("Tampilkan Dataset"):
            st.dataframe(df_iris)

if __name__ == "__main__":
    main()
