import streamlit as st
import numpy as np
import joblib

# === Configurações de página ===
st.set_page_config(page_title="Previsão de Obesidade", page_icon="🍔", layout="centered")

# === Carrega modelo e labels ===
modelo, scaler = joblib.load("modelo_obesidade_final.pkl")
labels = joblib.load("labels_obesidade.npy")

st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>🔍 Previsão de Obesidade</h1>
    <p style='text-align: center;'>Preencha o formulário abaixo para estimar sua categoria de obesidade com base em características comportamentais, físicas e sociais.</p>
    <hr style='border-top: 1px solid #bbb;'>
    """, unsafe_allow_html=True
)

# === Formulário ===
with st.form("form_obesidade"):
    col1, col2 = st.columns(2)

    with col1:
        genero = st.selectbox("Gênero", ["masculino", "feminino"])
        idade = st.slider("Idade", 5, 100, 30)
        peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, step=0.5)
        altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, step=0.01)
        historico_familiar = st.selectbox("Histórico familiar de sobrepeso?", ["sim", "não"])


    with col2:
        faf = st.number_input("Atividade Física por Semana (FAF em horas)", min_value=0, max_value=20, step=1)
        tue = st.number_input("Tempo Usando Dispositivos Eletrônicos (TUE em horas)", min_value=0, max_value=24, step=1)
        favc = st.selectbox("Consome comida calórica frequentemente (FAVC)?", ["sim", "não"])
        caec = st.selectbox("Frequência de lanche entre refeições (CAEC)", ["nunca", "às vezes", "frequentemente", "sempre"])
        calc = st.selectbox("Consumo de álcool (CALC)", ["nunca", "às vezes", "frequentemente", "sempre"])
        mtrans = st.selectbox("Meio de transporte", ["caminhada", "bicicleta", "moto", "automóvel", "transporte público"])
        
        
        

    submit = st.form_submit_button("📊 Prever")

# === Previsão ===
if submit:
    imc = peso / (altura ** 2)

    map_binario = {"sim": 1, "não": 0}
    map_ordem = {"nunca": 0, "às vezes": 1, "frequentemente": 2, "sempre": 3}
    map_transporte = {
        "caminhada": 0,
        "bicicleta": 0,
        "moto": 1,
        "automóvel": 2,
        "transporte público": 3
    }
    map_genero = {"masculino": 1, "feminino": 0}

    score_comport = (
        (map_binario[favc] == 1) +
        (map_ordem[caec] >= 2) +
        (map_ordem[calc] >= 2)
    )

    score_sedent = (
        (faf < 1) +
        (tue < 1) +
        (map_transporte[mtrans] in [1, 2])
    )

    risco_social = (
        (map_binario[historico_familiar] == 1) +
        (map_genero[genero] == 1) +
        (idade > 40)
    )

    entrada = np.array([[imc, score_comport, score_sedent, risco_social]])
    entrada_esc = scaler.transform(entrada)
    pred = modelo.predict(entrada_esc)[0]

    st.markdown("---")
    st.success(f"🧠 Resultado: Categoria de obesidade prevista é **{labels[pred]}**")
