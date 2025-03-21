import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title="Análise de Disponibilidade Eólica", layout="wide")

# Função para carregar os dados
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        df.columns = df.columns.astype(str)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

# Função para converter colunas corretamente
def preprocess_data(df):
    df = df.copy()
    df["Data"] = pd.to_datetime(df["Data"], errors='coerce')
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Função para calcular métricas principais
def calcular_metricas(df, disponibilidade_contratual, fator_multa, fator_bonus, pmd, mwh_anual):
    metricas = []
    
    df["Ano"] = df["Data"].dt.year  # Extrai o ano para agrupar
    parques = df["Windfarm"].unique()
    
    for parque in parques:
        df_parque = df[df["Windfarm"] == parque].groupby("Ano").mean(numeric_only=True)  # Agrupar por ano

        for ano, row in df_parque.iterrows():
            media_anual = row.mean()
            desvio_padrao = row.std()

            if media_anual < disponibilidade_contratual:
                multa = fator_multa * pmd * mwh_anual * ((disponibilidade_contratual / media_anual) - 1)
                bonus = 0
            else:
                bonus = fator_bonus * pmd * mwh_anual * ((disponibilidade_contratual / media_anual) - 1)
                multa = 0

            metricas.append({
                "Ano": ano,
                "Parque": parque,
                "Disponibilidade Média (%)": round(media_anual, 2),
                "Desvio Padrão": round(desvio_padrao, 2),
                "Multa (R$)": round(multa, 2),
                "Bônus (R$)": round(bonus, 2),
            })

    df_metricas = pd.DataFrame(metricas)
    
    # Adiciona os valores totais do projeto
    df_projeto = df_metricas.groupby("Ano").sum(numeric_only=True).reset_index()
    df_projeto["Parque"] = "Projeto (Total)"
    df_metricas = pd.concat([df_metricas, df_projeto])

    return df_metricas

# Função para análise probabilística
def analise_probabilistica(df, disponibilidade_contratual, valor_multa_usuario):
    df_prob = df.copy()
    
    if "Desvio Padrão" in df_prob.columns and df_prob["Desvio Padrão"].mean() > 0:
        df_prob["Probabilidade Abaixo do Contrato (%)"] = df_prob["Disponibilidade Média (%)"].apply(
            lambda x: round(stats.norm.cdf(disponibilidade_contratual, loc=x, scale=df_prob["Desvio Padrão"].mean()) * 100, 2)
        )
    else:
        df_prob["Probabilidade Abaixo do Contrato (%)"] = 0
    
    # Probabilidade de atingir um valor específico de multa para o projeto
    media_multa = df_prob[df_prob["Parque"] == "Projeto (Total)"]["Multa (R$)"].mean()
    desvio_multa = df_prob[df_prob["Parque"] == "Projeto (Total)"]["Multa (R$)"].std()

    if desvio_multa > 0:
        prob_multa = stats.norm.sf(valor_multa_usuario, loc=media_multa, scale=desvio_multa) * 100
    else:
        prob_multa = 0

    return df_prob, round(prob_multa, 2)

# Interface do Streamlit
st.title("📊 Análise de Disponibilidade de Parques Eólicos")

st.sidebar.header("Configurações")
uploaded_file = st.sidebar.file_uploader("📂 Faça o upload da planilha Excel", type=["xlsx"])

disponibilidade_contratual = st.sidebar.slider("Disponibilidade Contratual (%)", 80, 100, 95)
fator_multa = st.sidebar.number_input("Fator de Multa", min_value=0.0, value=1.5)
fator_bonus = st.sidebar.number_input("Fator de Bônus", min_value=0.0, value=1.2)
pmd = st.sidebar.number_input("PMD (R$/MWh)", min_value=0.0, value=300.0)
mwh_anual = st.sidebar.number_input("MWh por WTG/ano", min_value=0.0, value=700.0)
valor_multa_usuario = st.sidebar.number_input("Valor de multa esperado (R$)", min_value=0.0, value=500000.0)

if uploaded_file:
    df = load_data(uploaded_file)
    df = preprocess_data(df)

    if df is not None:
        st.write("### 🔍 Pré-visualização dos Dados")
        st.dataframe(df.head())

        # Cálculo de métricas
        df_metricas = calcular_metricas(df, disponibilidade_contratual, fator_multa, fator_bonus, pmd, mwh_anual)
        df_metricas, prob_multa = analise_probabilistica(df_metricas, disponibilidade_contratual, valor_multa_usuario)

        # Exibir métricas
        st.write("### 📈 Métricas Anuais dos Parques e Projeto")
        st.dataframe(df_metricas)

        # Gráfico de barras das multas e bônus por ano
        fig = px.bar(df_metricas[df_metricas["Parque"] == "Projeto (Total)"], 
                     x="Ano", y=["Multa (R$)", "Bônus (R$)"], 
                     title="Multa e Bônus Totais do Projeto por Ano",
                     labels={"value": "Valor (R$)", "Ano": "Ano"},
                     barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        # Probabilidade de atingir a multa inserida pelo usuário
        st.write(f"### 📊 Probabilidade de atingir uma multa de **R$ {valor_multa_usuario:,.2f}** no projeto:")
        st.write(f"📌 **{prob_multa:.2f}%** de chance de a multa total do projeto ser maior ou igual a esse valor.")

        # Probabilidade geral de ficar abaixo do contrato
        fig = px.bar(df_metricas, x="Parque", y="Probabilidade Abaixo do Contrato (%)",
                     title="Probabilidade de Disponibilidade ficar Abaixo do Contrato",
                     color="Parque")
        st.plotly_chart(fig, use_container_width=True)

        # Botão de download dos resultados
        csv_data = df_metricas.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar Resultados em CSV", data=csv_data, file_name="analise_projeto_eolico.csv", mime="text/csv")

else:
    st.warning("📌 Por favor, faça o upload de um arquivo Excel para iniciar a análise.")
