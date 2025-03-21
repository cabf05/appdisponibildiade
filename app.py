import streamlit as st
import pandas as pd
import numpy as np

# Configuração inicial do Streamlit
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

# Função para pré-processar os dados
def preprocess_data(df):
    df = df.copy()
    # Converter colunas de data para valores numéricos (disponibilidade)
    date_columns = [col for col in df.columns if col not in ['Windfarm', 'WTGs']]
    df[date_columns] = df[date_columns].apply(pd.to_numeric, errors='coerce')
    return df

# Função para calcular métricas principais
def calcular_metricas(df, disponibilidade_contratual, fator_multa, fator_bonus, pmd, mwh_por_wtg_ano):
    metricas = []
    total_multa_projeto = 0
    total_bonus_projeto = 0

    # Identificar anos nas colunas de data
    date_columns = [col for col in df.columns if col not in ['Windfarm', 'WTGs']]
    df_dates = pd.to_datetime(date_columns)
    anos = df_dates.year.unique()

    for ano in anos:
        colunas_ano = [col for col in date_columns if pd.to_datetime(col).year == ano]
        for parque in df['Windfarm'].unique():
            df_parque = df[df['Windfarm'] == parque]
            wtgs = df_parque['WTGs'].iloc[0]
            disponibilidades = df_parque[colunas_ano].mean(axis=1).iloc[0]
            media_anual = disponibilidades

            mwh_anual_parque = mwh_por_wtg_ano * wtgs

            if media_anual < disponibilidade_contratual:
                multa = fator_multa * pmd * mwh_anual_parque * ((disponibilidade_contratual / media_anual) - 1)
                bonus = 0
            else:
                bonus = fator_bonus * pmd * mwh_anual_parque * ((media_anual / disponibilidade_contratual) - 1)
                multa = 0

            total_multa_projeto += multa
            total_bonus_projeto += bonus

            metricas.append({
                "Parque": parque,
                "Ano": ano,
                "Disponibilidade Média Anual (%)": round(media_anual, 2),
                "MWh Anual": round(mwh_anual_parque, 2),
                "Multa (R$)": round(multa, 2),
                "Bônus (R$)": round(bonus, 2),
            })

    df_metricas = pd.DataFrame(metricas)
    df_metricas["Total Multa Projeto (R$)"] = round(total_multa_projeto, 2)
    df_metricas["Total Bônus Projeto (R$)"] = round(total_bonus_projeto, 2)
    
    return df_metricas

# Função para calcular probabilidade empírica de multa do projeto
def calcular_prob_empirica_multa(df_metricas, valor_multa_desejado):
    multas_anuais = df_metricas.groupby('Ano')['Multa (R$)'].sum()
    prob = (multas_anuais > valor_multa_desejado).mean()
    return prob

# Interface do Streamlit
st.title("📊 Análise de Disponibilidade de Parques Eólicos")

# Sidebar para configurações
st.sidebar.header("Configurações")
uploaded_file = st.sidebar.file_uploader("📂 Faça o upload da planilha Excel", type=["xlsx"])
disponibilidade_contratual = st.sidebar.slider("Disponibilidade Contratual (%)", 80, 100, 95)
fator_multa = st.sidebar.number_input("Fator de Multa", min_value=0.0, value=1.5)
fator_bonus = st.sidebar.number_input("Fator de Bônus", min_value=0.0, value=1.2)
pmd = st.sidebar.number_input("PMD (R$/MWh)", min_value=0.0, value=300.0)
mwh_por_wtg_ano = st.sidebar.number_input("MWh por WTG por Ano", min_value=0.0, value=730.0)
valor_multa_desejado = st.sidebar.number_input("Valor de Multa para Probabilidade (R$)", min_value=0.0, value=100000.0)

# Processamento após upload
if uploaded_file:
    df = load_data(uploaded_file)
    df = preprocess_data(df)

    if df is not None:
        st.write("### 🔍 Pré-visualização dos Dados")
        st.dataframe(df.head())

        # Cálculo de métricas
        df_metricas = calcular_metricas(df, disponibilidade_contratual, fator_multa, fator_bonus, pmd, mwh_por_wtg_ano)

        # Exibir métricas
        st.write("### 📈 Métricas dos Parques Eólicos")
        st.dataframe(df_metricas)

        # Total do projeto
        total_multa = df_metricas["Total Multa Projeto (R$)"].iloc[0]
        total_bonus = df_metricas["Total Bônus Projeto (R$)"].iloc[0]
        st.write(f"#### Total Anual do Projeto")
        st.write(f"- **Multa Total**: R$ {total_multa:,.2f}")
        st.write(f"- **Bônus Total**: R$ {total_bonus:,.2f}")

        # Explicação do cálculo de multa e bônus
        st.write("### Cálculo de Multa e Bônus")
        st.write("""
        - **Multa**: Se a disponibilidade média anual do parque for menor que a disponibilidade contratual, a multa é calculada como:
        \[
        \text{Multa} = \text{fator_multa} \times \text{PMD} \times \text{MWh_anual} \times \left( \frac{\text{disponibilidade_contratual}}{\text{disponibilidade_média}} - 1 \right)
        \]
        - **Bônus**: Se a disponibilidade média anual do parque for maior ou igual à disponibilidade contratual, o bônus é calculado como:
        \[
        \text{Bônus} = \text{fator_bônus} \times \text{PMD} \times \text{MWh_anual} \times \left( \frac{\text{disponibilidade_média}}{\text{disponibilidade_contratual}} - 1 \right)
        \]
        """)

        # Probabilidade empírica de multa do projeto
        prob_multa = calcular_prob_empirica_multa(df_metricas, valor_multa_desejado)
        st.write(f"### Probabilidade de Multa Total do Projeto Superior a R$ {valor_multa_desejado:,.2f}")
        st.write(f"Probabilidade: **{prob_multa * 100:.2f}%**")
        st.write("*Esta probabilidade é calculada com base na frequência histórica de anos em que a soma das multas dos parques ultrapassou o valor especificado.*")

else:
    st.warning("📌 Por favor, faça o upload de um arquivo Excel para iniciar a análise.")
