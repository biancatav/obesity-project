import streamlit as st
import joblib
import numpy as np
import pandas as pd  # ← essa linha é a correção!

st.set_page_config(page_title="Previsão de Obesidade", layout="centered")

# Título
st.title("🔍 Previsão de Nível de Obesidade")
st.markdown("Preencha os dados abaixo para prever o nível de obesidade:")

# Campos de entrada (customize conforme suas features)
gender = st.selectbox("Gênero", ["Male", "Female"])
age = st.slider("Idade", 10, 80, 25)
height = st.number_input("Altura (em metros)", min_value=1.0, max_value=2.5, value=1.70)
weight = st.number_input("Peso (em kg)", min_value=30.0, max_value=200.0, value=70.0)
favc = st.selectbox("Consome alimentos altamente calóricos com frequência?", ["yes", "no"])
fcvc = st.slider("Frequência de consumo de vegetais (0-3)", 0.0, 3.0, 2.0)
ncp = st.slider("Número de refeições por dia", 1.0, 6.0, 3.0)
caec = st.selectbox("Consome alimentos entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Fuma?", ["yes", "no"])
scc = st.selectbox("Segue conselhos de profissionais da saúde?", ["yes", "no"])
calc = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently"])
ch2o = st.slider("Consumo diário de água (litros)", 0.0, 5.0, 2.0)
faf = st.slider("Frequência de atividade física", 0.0, 3.0, 1.0)
tue = st.slider("Tempo de uso de tecnologia (horas por dia)", 0.0, 5.0, 2.0)

# Feature Engineering simplificada
bmi = weight / (height ** 2)
risk_score = int(favc == "yes") + int(caec in ["Always", "Frequently"]) + \
             int(calc == "Frequently") + int(scc == "no") + int(smoke == "yes")
water_per_kg = ch2o / weight
active_vs_sedentary = faf - tue

# Montar DataFrame com dados para predição
df_input = pd.DataFrame([{
    "Gender": 1 if gender == "Male" else 0,
    "Age": age,
    "Height": height,
    "Weight": weight,
    "FAVC": 1 if favc == "yes" else 0,
    "FCVC": fcvc,
    "NCP": ncp,
    "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[caec],
    "SMOKE": 1 if smoke == "yes" else 0,
    "CH2O": ch2o,
    "SCC": 1 if scc == "yes" else 0,
    "FAF": faf,
    "TUE": tue,
    "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2}[calc],
    "BMI": bmi,
    "risk_score": risk_score,
    "water_per_kg": water_per_kg,
    "active_vs_sedentary": active_vs_sedentary
}])

# Botão de previsão
if st.button("🔮 Prever"):
    pred = model.predict(df_input)[0]
    st.success(f"✅ Previsão: **{pred}**")