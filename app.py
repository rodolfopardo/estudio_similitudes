import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Título de la app
st.title("Estudio de Similitud de Keywords con Títulos o Textos - Big Hacks")

# **Seleccionar idioma para las stopwords**
st.sidebar.header("Configuración")
stopwords_language = st.sidebar.selectbox(
    "Selecciona el idioma para las stopwords",
    options=["english", "spanish"],
    index=0
)

# **Carga de Archivos CSV**
st.header("Carga de archivos")
queries_file = st.file_uploader("Sube tu archivo de Keywords (CSV)", type="csv")
urls_file = st.file_uploader("Sube tu archivo de URLs o Títulos (CSV)", type="csv")

if queries_file and urls_file:
    # **1. Leer archivos CSV**
    queries_df = pd.read_csv(queries_file)
    urls_df = pd.read_csv(urls_file)
    
    # **2. Procesar los datos**
    try:
        queries = queries_df['Queries'].tolist()
        urls = urls_df['URLs'].tolist()
    except KeyError:
        st.error("Asegúrate de que los archivos tengan las columnas 'Queries' y 'URLs'.")
    else:
        # **3. Vectorización con TF-IDF**
        texts = queries + urls
        vectorizer = TfidfVectorizer(stop_words=stopwords_language)
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Dividir las matrices
        queries_vectors = tfidf_matrix[:len(queries)]
        urls_vectors = tfidf_matrix[len(queries):]

        # **4. Calcular similitud coseno**
        similarity_matrix = cosine_similarity(queries_vectors, urls_vectors)

        # **5. Calcular coincidencia promedio por query**
        average_similarity = similarity_matrix.mean(axis=1)

        # Crear DataFrame con resultados
        results_df = pd.DataFrame({
            'Query': queries,
            'Coincidencia_Promedio': average_similarity
        })

        # Ordenar y seleccionar las 50 queries principales
        top_50_queries = results_df.sort_values(by='Coincidencia_Promedio', ascending=False).head(50)

        # **6. Mostrar resultados**
        st.header("Top 50 Queries con Mayor Coincidencia Promedio")
        st.dataframe(top_50_queries)

        # **Visualización**
        st.header("Visualización de las Top 20 Queries")
        top_20 = top_50_queries.head(20)

        # Crear un gráfico de barras
        plt.figure(figsize=(10, 8))
        plt.barh(top_20['Query'], top_20['Coincidencia_Promedio'], color='skyblue', edgecolor='black')
        plt.xlabel("Coincidencia Promedio", fontsize=12)
        plt.ylabel("Query", fontsize=12)
        plt.title("Top 20 Queries por Coincidencia Promedio", fontsize=14)
        plt.gca().invert_yaxis()  # Invertir el eje Y para mejor lectura
        plt.tight_layout()

        # Mostrar el gráfico en Streamlit
        st.pyplot(plt)

        # Opción para descargar el archivo de resultados
        st.header("Descargar Resultados")
        csv = top_50_queries.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar CSV de las Top 50 Queries",
            data=csv,
            file_name="top_50_queries.csv",
            mime="text/csv"
        )
