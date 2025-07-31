import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# === Configurações ===
st.set_page_config(page_title="Dashboard Obesidade", page_icon="📈", layout="wide")

st.title("📊 Dashboard Analítico - Obesidade")

# === Carregamento de dados ===
@st.cache_data
def carregar_dados():
    df = pd.read_csv("Obesity.csv")
    df.rename(columns={'Obesity': 'NObeyesdad', 'family_history': 'family_history_with_overweight'}, inplace=True)
    return df

df = carregar_dados()

# === Limpeza básica ===
df['Gender'] = df['Gender'].str.strip().str.capitalize()
df['NObeyesdad'] = df['NObeyesdad'].str.strip()

# === Layout ===
aba = st.sidebar.radio("🔎 Selecione a Análise:", [
    "Visão Geral",
    "Distribuição por Gênero",
    "Distribuição por Categoria de Obesidade",
    "Correlação entre Variáveis Numéricas",
    "Análise Interativa"
])

# === Visão Geral ===
if aba == "Visão Geral":
    st.subheader("📌 Informações Gerais")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de Registros", df.shape[0])
    with col2:
        st.metric("Total de Categorias", df['NObeyesdad'].nunique())

    st.markdown("---")
    st.write("### Distribuição por Categoria")
    st.bar_chart(df['NObeyesdad'].value_counts())

# === Gênero ===
elif aba == "Distribuição por Gênero":
    st.subheader("📊 Distribuição por Gênero")
    genero_count = df['Gender'].value_counts()
    fig = px.pie(names=genero_count.index, values=genero_count.values, title="Gênero na Amostra", hole=0.3)
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Obesidade por Gênero")
    fig2 = px.histogram(df, x="NObeyesdad", color="Gender", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

# === Categoria de Obesidade ===
elif aba == "Distribuição por Categoria de Obesidade":
    st.subheader("🏷️ Categorias de Obesidade")
    fig = px.histogram(df, x="NObeyesdad", color="NObeyesdad", title="Frequência por Categoria")
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Relação com Consumo Calórico (FAVC)")
    favc_map = df['FAVC'].str.lower().map({'yes': 'Sim', 'no': 'Não'})
    fig = px.histogram(df, x="NObeyesdad", color=favc_map, barmode="group", title="FAVC x Obesidade")
    st.plotly_chart(fig, use_container_width=True)

# === Correlação ===
elif aba == "Correlação entre Variáveis Numéricas":
    st.subheader("📈 Mapa de Correlação")
    campos_numericos = ['Age', 'Height', 'Weight', 'FAF', 'TUE']
    corr = df[campos_numericos].corr()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)

# === Interativo ===
elif aba == "Análise Interativa":
    st.subheader("🎯 Explore os Dados")

    col1, col2 = st.columns(2)
    with col1:
        eixo_x = st.selectbox("Eixo X", df.select_dtypes(include='number').columns)
    with col2:
        eixo_y = st.selectbox("Eixo Y", df.select_dtypes(include='number').columns, index=1)

    fig = px.scatter(df, x=eixo_x, y=eixo_y, color="NObeyesdad", hover_data=["Gender", "Age"])
    st.plotly_chart(fig, use_container_width=True)