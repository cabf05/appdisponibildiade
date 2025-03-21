import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats

st.set_page_config(page_title="Análise de Disponibilidade Eólica", layout="wide")

# Função para carregar os dados
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        df.columns = df.columns.astype(str).str.strip()  # Remove espaços extras nos nomes das colunas
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

# Função para processar os dados
def preprocess_data(df):
    df = df.copy()

    # Verifica e padroniza a coluna de Data
    possible_date_columns = ["Data", "data", "DATE", "date", "Data "]
    date_column = next((col for col in possible_date_columns if col in df.columns), None)

    if not date_column:
        st.error("Erro: A coluna de Data não foi encontrada no arquivo. Verifique o cabeçalho do arquivo Excel.")
        return None

    df["Data"] = pd.to_datetime(df[date_column], errors='coerce')

    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# Cálculo das médias anuais e valores do projeto
def calcular_estatisticas(df, fator_multa, fator_bonus, pmd, mwh_anual, disponibilidade_contrato):
    df["Ano"] = df["Data"].dt.year
    medias_anuais = df.groupby("Ano").mean().reset_index()

    # Cálculo de multas e bônus
    medias_anuais["Multa"] = np.where(
        medias_anuais["Disponibilidade"] < disponibilidade_contrato,
        fator_multa * pmd * mwh_anual * ((disponibilidade_contrato / medias_anuais["Disponibilidade"]) - 1),
        0
    )

    medias_anuais["Bônus"] = np.where(
        medias_anuais["Disponibilidade"] > disponibilidade_contrato,
        fator_bonus * pmd * mwh_anual * ((disponibilidade_contrato / medias_anuais["Disponibilidade"]) - 1),
        0
    )

    # Valor total do projeto (soma de multas e bônus)
    valor_total_projeto = medias_anuais["Multa"].sum() + medias_anuais["Bônus"].sum()

    return medias_anuais, valor_total_projeto

# Cálculo da probabilidade de atingir um determinado valor de multa
def calcular_probabilidade(medias_anuais, valor_multa_alvo):
    media_multa = medias_anuais["Multa"].mean()
    desvio_multa = medias_anuais["Multa"].std()

    if desvio_multa == 0:
        return 0  # Evita divisão por zero

    prob = 1 - stats.norm.cdf(valor_multa_alvo, loc=media_multa, scale=desvio_multa)
    return prob

# Interface do Streamlit
st.title("📊 Análise de Disponibilidade de Parques Eólicos")

st.sidebar.header("Configurações")
uploaded_file = st.sidebar.file_uploader("📂 Faça o upload da planilha Excel", type=["xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        df = preprocess_data(df)
        if df is not None:
            st.write("### 🔍 Pré-visualização dos Dados")
            st.dataframe(df.head())

            # Entrada dos parâmetros para cálculo
            fator_multa = st.sidebar.number_input("Fator de Multa", min_value=0.0, value=0.1, step=0.01)
            fator_bonus = st.sidebar.number_input("Fator de Bônus", min_value=0.0, value=0.1, step=0.01)
            pmd = st.sidebar.number_input("PMD", min_value=0.0, value=100.0, step=1.0)
            mwh_anual = st.sidebar.number_input("MWh Anual", min_value=0.0, value=50000.0, step=1000.0)
            disponibilidade_contrato = st.sidebar.number_input("Disponibilidade Contratual", min_value=0.0, max_value=1.0, value=0.95, step=0.01)

            # Cálculo das estatísticas
            medias_anuais, valor_total_projeto = calcular_estatisticas(df, fator_multa, fator_bonus, pmd, mwh_anual, disponibilidade_contrato)

            # Gráfico de médias anuais de disponibilidade
            st.write("## 📈 Média Anual de Disponibilidade")
            fig = px.line(medias_anuais, x="Ano", y="Disponibilidade", markers=True, title="Média Anual de Disponibilidade")
            st.plotly_chart(fig)

            # Gráfico de multas e bônus anuais
            st.write("## 💰 Multas e Bônus Anuais")
            fig2 = px.bar(medias_anuais, x="Ano", y=["Multa", "Bônus"], title="Multas e Bônus Anuais", barmode="group")
            st.plotly_chart(fig2)

            # Valor total do projeto
            st.write(f"### 💰 Valor Total do Projeto: **R$ {valor_total_projeto:,.2f}**")

            # Probabilidade de atingir um determinado valor de multa
            valor_multa_alvo = st.sidebar.number_input("Insira um valor de multa alvo", min_value=0.0, value=100000.0, step=5000.0)
            prob_multa = calcular_probabilidade(medias_anuais, valor_multa_alvo)
            st.write(f"### 🎯 Probabilidade de atingir R$ {valor_multa_alvo:,.2f} em multas: **{prob_multa:.2%}**")

        else:
            st.warning("⚠️ O arquivo não contém uma coluna de Data válida.")
else:
    st.warning("📌 Por favor, faça o upload de um arquivo Excel para iniciar a análise.")
