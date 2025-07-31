import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np

st.set_page_config(page_title="Relatório de Obesidade", layout="wide")

# === Carregamento de dados ===
@st.cache_data
def carregar_dados():
    df = pd.read_csv("Obesity.csv")
    df.rename(columns={'Obesity': 'NObeyesdad', 'family_history': 'family_history_with_overweight'}, inplace=True)
    return df

df = carregar_dados()

df.rename(columns={"NObeyesdad": "Obesity"}, inplace=True)

# Título
st.title("📘 Relatório Analítico e Estratégico – Prevenção e Diagnóstico de Obesidade")

# Tabs de navegação
aba1, aba2 = st.tabs(["🔬 Visão Analítica (Data Science)", "🏥 Visão Clínica e Estratégica"])

# === ABA 1 === #
with aba1:
    st.header("🔬 Análise Técnica e Justificativa do Modelo")

    st.markdown("### 📌 Descrição do Dataset")
    st.markdown("""
    O dataset contém **2.111 registros** de pacientes com informações demográficas, comportamentais e clínicas.
    A variável alvo é **Obesity**, classificada em categorias como *Normal*, *Overweight*, *Obesity Type I*, etc.
    """)

    st.dataframe(df.head())

    st.markdown("### 📊 Distribuição das Classes Alvo")
    fig = px.histogram(df, x="Obesity", color="Obesity")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ⚙️ Feature Engineering")
    st.markdown("""
    Foram desenvolvidas variáveis compostas para melhorar a capacidade explicativa do modelo, como por exemplo:
    - `IMC`: Índice de Massa Corporal.
    - `score_comport`: comportamento alimentar e consumo.
    - `score_sedent`: hábitos sedentários e baixa ingestão de água.
    - `risco_social`: combinação de idade com sedentarismo.
    """)

    st.markdown("### 🤖 Modelo Utilizado: Random Forest Classifier")
    st.markdown("""
    O modelo escolhido foi o **Random Forest**, por apresentar:

    - Excelente performance em problemas de classificação com dados tabulares.
    - Capacidade de lidar bem com **dados heterogêneos** (variáveis numéricas e categóricas).
    - **Robustez contra overfitting** devido ao uso de múltiplas árvores de decisão com amostragem aleatória (bagging).
    - Possui **explicabilidade nativa** via importância das variáveis.
    - Funciona bem mesmo com dados que apresentam ruído ou relações não lineares.

    Por essas razões, o Random Forest foi selecionado como uma solução robusta e confiável para prever os níveis de obesidade com base em múltiplos fatores comportamentais e clínicos.
    """)


    st.markdown("### 🧪 Métricas de Avaliação (em validação):")
    st.write("- **Acurácia:** ~91%")
    st.write("- **ROC AUC:** ~0.95")
    st.write("- **Precisão/Recall balanceado** em todas as classes")

    st.markdown("### 🔥 Importância das Features")
    try:
        importances = pd.read_csv("feature_importances.csv")
        fig = px.bar(importances, x="feature", y="importance", title="Importância das Variáveis no Modelo")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Arquivo com importâncias de features não encontrado.")

    st.markdown("### 🧬 Mapa de Correlação das Variáveis Numéricas")
    corr = df.select_dtypes(include='number').corr()
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", ax=ax)
    st.pyplot(fig_corr)

# === ABA 2 === #
with aba2:
    st.header("🏥 Visão Estratégica para Equipe Médica")

    st.markdown("""
    Este painel visa **traduzir os resultados da modelagem de dados em decisões práticas** para prevenção e acompanhamento da obesidade.  
    A ferramenta pode ser usada para:
    - Identificar **pacientes com maior risco** a partir de seus hábitos.
    - Suportar triagem preventiva em check-ups clínicos.
    - Reforçar a atuação em **educação nutricional e mudanças de estilo de vida**.
    """)

    st.markdown("### 🧭 Perfis de Risco por Comportamento")
    fig = px.box(df, x="Obesity", y="FAF", color="Obesity", title="Atividade Física por Nível de Obesidade")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(df, x="Obesity", y="CH2O", color="Obesity", title="Ingestão de Água por Nível de Obesidade")
    st.plotly_chart(fig, use_container_width=True)

    # st.markdown("### 🧠 Predição Individual com Apoio de IA")

    # with st.form("predict_form"):
    #     gender = st.selectbox("Gênero", ["Male", "Female"])
    #     age = st.slider("Idade", 10, 100, 25)
    #     height = st.number_input("Altura (m)", 1.0, 2.5, step=0.01)
    #     weight = st.number_input("Peso (kg)", 30.0, 200.0, step=0.5)
    #     favc = st.selectbox("Consome comida altamente calórica?", ["yes", "no"])
    #     fcvc = st.slider("Frequência de vegetais (0-3)", 0.0, 3.0, 2.0)
    #     ncp = st.slider("Número de refeições principais", 1, 4, 3)
    #     caec = st.selectbox("Petisca entre refeições?", ["no", "Sometimes", "Frequently", "Always"])
    #     ch2o = st.slider("Copos de água por dia", 0.0, 3.0, 2.0)
    #     faf = st.slider("Atividade física por semana (horas)", 0.0, 10.0, 2.0)
    #     submit = st.form_submit_button("Prever Risco")

    # if submit:
    #     input_dict = {
    #         "Gender": 1 if gender == "Male" else 0,
    #         "Age": age,
    #         "Height": height,
    #         "Weight": weight,
    #         "FAVC": 1 if favc == "yes" else 0,
    #         "FCVC": fcvc,
    #         "NCP": ncp,
    #         "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[caec],
    #         "CH2O": ch2o,
    #         "FAF": faf
    #     }

    #     try:
    #         model, scaler = joblib.load("modelo_obesidade_final.pkl")
    #         input_df = pd.DataFrame([input_dict])

    #         # Calcula variáveis compostas
    #         input_df["IMC"] = input_df["Weight"] / (input_df["Height"] ** 2)
    #         input_df["score_comport"] = (
    #             input_df["FAVC"] +
    #             np.where(input_df["CAEC"] >= 2, 1, 0) +
    #             np.where(input_df["FCVC"] < 1.5, 1, 0)
    #         )
    #         input_df["score_sedent"] = (
    #             np.where(input_df["FAF"] < 1.0, 1, 0) +
    #             np.where(input_df["CH2O"] < 1.0, 1, 0)
    #         )
    #         input_df["risco_social"] = (
    #             np.where((input_df["Age"] >= 35) & (input_df["FAF"] == 0), 1, 0)
    #         )

    #         X_final = input_df[["IMC", "score_comport", "score_sedent", "risco_social"]]
    #         X_scaled = scaler.transform(X_final)
    #         prediction = model.predict(X_scaled)[0]
    #         st.success(f"🔎 Resultado predito: **{prediction}**")
    #     except Exception as e:
    #         st.error(f"Erro ao processar predição: {e}")
