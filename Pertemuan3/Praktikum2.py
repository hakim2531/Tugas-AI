import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klasifikasi Iris Link", layout="wide")

# --- LOAD DATASET DARI LINK ---
@st.cache_data
def load_data_from_url():
    # Menggunakan link raw dataset iris dari GitHub Seaborn
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url)
    return df

# --- TRAINING MODEL ---
def train_model(df):
    # Fitur: sepal_length, sepal_width, petal_length, petal_width
    # Target: species
    X = df.drop(columns=['species'])
    y = df['species']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

def main():
    st.title("🌸 Aplikasi Klasifikasi Iris (Data via URL)")
    st.write("Aplikasi ini mengambil data langsung dari link CSV online.")

    df_iris = load_data_from_url()
    
    if df_iris is not None:
        model, accuracy = train_model(df_iris)

        # --- SIDEBAR UNTUK INPUT ---
        st.sidebar.header("📝 Input Spesifikasi Bunga")
        
        # Penyesuaian nama variabel slider sesuai kolom CSV dari link
        sl = st.sidebar.slider("Sepal Length (cm)", float(df_iris['sepal_length'].min()), float(df_iris['sepal_length'].max()), 5.1)
        sw = st.sidebar.slider("Sepal Width (cm)", float(df_iris['sepal_width'].min()), float(df_iris['sepal_width'].max()), 3.5)
        pl = st.sidebar.slider("Petal Length (cm)", float(df_iris['petal_length'].min()), float(df_iris['petal_length'].max()), 1.4)
        pw = st.sidebar.slider("Petal Width (cm)", float(df_iris['petal_width'].min()), float(df_iris['petal_width'].max()), 0.2)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🔮 Hasil Prediksi")
            if st.button("Prediksi Sekarang"):
                # Data input harus berupa 2D array
                input_data = np.array([[sl, sw, pl, pw]])
                prediksi = model.predict(input_data)[0]
                st.success(f"Spesies Terdeteksi: **{prediksi}**")
            
            st.write("---")
            st.subheader("📊 Evaluasi Model")
            st.metric(label="Akurasi Model", value=f"{accuracy * 100:.2f}%")

        with col2:
            st.subheader("📈 Visualisasi Posisi Data")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.scatterplot(data=df_iris, x='sepal_length', y='petal_length', hue='species', ax=ax)
            
            # Menandai posisi input user
            ax.scatter(sl, pl, color='red', marker='X', s=200, label='Input User')
            ax.set_title("Sepal Length vs Petal Length")
            ax.legend()
            st.pyplot(fig)

        if st.checkbox("Tampilkan Tabel Data dari URL"):
            st.dataframe(df_iris, use_container_width=True)

if __name__ == "__main__":
    main()
