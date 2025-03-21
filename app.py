import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configuração inicial do Streamlit
st.set_page_config(page_title="Análise de Parques Eólicos", layout="wide")
st.title("Análise de Disponibilidade de Parques Eólicos")

# Função para carregar e validar os dados
@st.cache_data
def load_data(file):
    try:
        df = pd.read_excel(file)
        required_columns = ['Windfarm', 'WTGs']
        if not all(col in df.columns for col in required_columns):
            st.error("O arquivo deve conter as colunas 'Windfarm' e 'WTGs'.")
            return None
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

# Função para calcular médias móveis
@st.cache_data
def calculate_moving_average(df, window):
    date_cols = [col for col in df.columns if col not in ['Windfarm', 'WTGs']]
    df_numeric = df[date_cols].apply(pd.to_numeric, errors='coerce')
    return df_numeric.rolling(window=window, axis=1).mean()

# Função para calcular impacto financeiro simplificado
def calculate_financial_impact(df, contractual_availability, penalty_factor, pmd, mwh_per_wtg_day):
    date_cols = [col for col in df.columns if col not in ['Windfarm', 'WTGs']]
    df_numeric = df[date_cols].apply(pd.to_numeric, errors='coerce')
    annual_mean = df_numeric.mean(axis=1).iloc[0]
    wtgs = df['WTGs'].iloc[0]
    diff = annual_mean - contractual_availability
    if diff < 0:
        loss_mwh = abs(diff) * wtgs * mwh_per_wtg_day * 365
        penalty = loss_mwh * pmd * penalty_factor
        return -penalty, loss_mwh
    return 0, 0

# Interface do usuário
with st.sidebar:
    st.header("Parâmetros de Entrada")
    uploaded_file = st.file_uploader("Carregar Planilha Excel", type=["xlsx"])
    penalty_factor = st.number_input("Fator Multa", min_value=0.0, value=1.0, step=0.1)
    contractual_availability = st.number_input("Disponibilidade Contratual (%)", min_value=0.0, max_value=100.0, value=95.0) / 100
    pmd = st.number_input("PMD (R$/MWh)", min_value=0.0, value=300.0, step=10.0)
    mwh_per_wtg_day = st.number_input("MWh por WTG/dia", min_value=0.0, value=10.0, step=1.0)

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        windfarms = df['Windfarm'].unique()
        selected_windfarm = st.sidebar.selectbox("Selecione o Parque Eólico", windfarms)
        df_selected = df[df['Windfarm'] == selected_windfarm].drop(columns=['Windfarm'])

        # Abas da interface
        tab1, tab2, tab3 = st.tabs(["Visão Anual", "Análise Temporal", "Impacto Financeiro"])

        with tab1:
            st.subheader("Visão Anual")
            date_cols = [col for col in df_selected.columns if col not in ['WTGs']]
            annual_mean = df_selected[date_cols].mean(axis=1).iloc[0]
            st.write(f"Média Anual de Disponibilidade: {annual_mean:.2%}")
            stats_desc = df_selected[date_cols].describe().T
            st.dataframe(stats_desc)

        with tab2:
            st.subheader("Análise Temporal")
            window = st.slider("Janela da Média Móvel (meses)", min_value=1, max_value=12, value=3)
            moving_avg = calculate_moving_average(df_selected, window)
            fig = px.line(x=date_cols, y=moving_avg.iloc[0], title="Série Temporal com Média Móvel")
            fig.add_hline(y=contractual_availability, line_dash="dash", line_color="red", annotation_text="Contrato")
            st.plotly_chart(fig)

        with tab3:
            st.subheader("Impacto Financeiro")
            financial_impact, energy_loss = calculate_financial_impact(df_selected, contractual_availability, penalty_factor, pmd, mwh_per_wtg_day)
            st.write(f"Impacto Financeiro Anual: R${financial_impact:,.2f}")
            st.write(f"Perda Energética Anual: {energy_loss:,.2f} MWh")
            fig_corr = px.scatter(x=df_selected[date_cols].iloc[0], y=[financial_impact]*len(date_cols), title="Disponibilidade x Impacto")
            st.plotly_chart(fig_corr)

        # Download de resultados
        csv = df_selected.to_csv(index=False)
        st.download_button("Baixar Resultados em CSV", csv, "resultados.csv", "text/csv")
else:
    st.info("Por favor, carregue uma planilha Excel para começar a análise.")
