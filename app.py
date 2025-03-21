import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import openpyxl

# Configuração inicial do Streamlit
st.set_page_config(page_title="Análise de Disponibilidade de Parques Eólicos", layout="wide")
st.title("Análise Avançada de Disponibilidade de Parques Eólicos")

# Função para carregar e validar os dados
@st.cache_data
def load_data(file):
    try:
        df = pd.read_excel(file)
        required_columns = ['Windfarm', 'WTGs']
        if not all(col in df.columns for col in required_columns):
            st.error("O arquivo deve conter as colunas 'Windfarm' e 'WTGs'.")
            return None
        # Verificar colunas de datas no formato MM/DD/AAAA
        date_cols = [col for col in df.columns if col not in required_columns]
        for col in date_cols:
            pd.to_datetime(df[col], format='%m/%d/%Y', errors='coerce')
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

# Função para ajustar distribuições e selecionar a melhor
@st.cache_data
def fit_distribution(data):
    distributions = [stats.norm, stats.lognorm, stats.beta, stats.weibull_min]
    results = {}
    for dist in distributions:
        if dist == stats.beta:
            params = dist.fit(data, floc=0, fscale=1)  # Ajuste para beta entre 0 e 1
        elif dist == stats.weibull_min:
            params = dist.fit(data, floc=0)  # Localização fixa em 0
        else:
            params = dist.fit(data)
        log_likelihood = dist.logpdf(data, *params).sum()
        results[dist.__name__] = {'params': params, 'log_likelihood': log_likelihood}
    best_dist = max(results.items(), key=lambda x: x[1]['log_likelihood'])
    return best_dist[0], best_dist[1]['params']

# Função para calcular multas/bônus
def calculate_financial_impact(df, contractual_availability, penalty_factor, bonus_factor, pmd, mwh_per_wtg_day):
    date_cols = [col for col in df.columns if col not in ['Windfarm', 'WTGs']]
    df_numeric = df[date_cols].apply(pd.to_numeric, errors='coerce')
    annual_means = df_numeric.mean(axis=1)
    wtgs = df['WTGs'].iloc[0]
    financial_impact = []
    energy_loss = []
    for mean in annual_means:
        diff = mean - contractual_availability
        if diff < 0:
            # Multa
            loss_mwh = abs(diff) * wtgs * mwh_per_wtg_day * 365
            penalty = loss_mwh * pmd * penalty_factor
            financial_impact.append(-penalty)
            energy_loss.append(loss_mwh)
        else:
            # Bônus
            gain_mwh = diff * wtgs * mwh_per_wtg_day * 365
            bonus = gain_mwh * pmd * bonus_factor
            financial_impact.append(bonus)
            energy_loss.append(-gain_mwh)
    return financial_impact, energy_loss

# Interface do usuário
with st.sidebar:
    st.header("Parâmetros de Entrada")
    uploaded_file = st.file_uploader("Carregar Planilha Excel", type=["xlsx"])
    penalty_factor = st.number_input("Fator Multa", min_value=0.0, value=1.0, step=0.1)
    contractual_availability = st.number_input("Disponibilidade Contratual (%)", min_value=0.0, max_value=100.0, value=95.0) / 100
    bonus_factor = st.number_input("Fator Bônus", min_value=0.0, value=0.5, step=0.1)
    pmd = st.number_input("PMD (R$/MWh)", min_value=0.0, value=300.0, step=10.0)
    mwh_per_wtg_day = st.number_input("MWh por WTG/dia", min_value=0.0, value=10.0, step=1.0)

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        windfarms = df['Windfarm'].unique()
        selected_windfarm = st.sidebar.selectbox("Selecione o Parque Eólico", windfarms)
        df_selected = df[df['Windfarm'] == selected_windfarm].drop(columns=['Windfarm'])

        # Abas da interface
        tab1, tab2, tab3, tab4 = st.tabs(["Visão Anual", "Análise Temporal", "Impacto Energético", "Modelagem Probabilística"])

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

            # Decomposição sazonal
            result = seasonal_decompose(df_selected[date_cols].iloc[0], model='additive', period=12)
            fig_decomp = go.Figure()
            fig_decomp.add_trace(go.Scatter(x=date_cols, y=result.trend, mode='lines', name='Tendência'))
            fig_decomp.add_trace(go.Scatter(x=date_cols, y=result.seasonal, mode='lines', name='Sazonalidade'))
            fig_decomp.add_trace(go.Scatter(x=date_cols, y=result.resid, mode='lines', name='Resíduos'))
            st.plotly_chart(fig_decomp)

        with tab3:
            st.subheader("Impacto Energético e Financeiro")
            financial_impact, energy_loss = calculate_financial_impact(df_selected, contractual_availability, penalty_factor, bonus_factor, pmd, mwh_per_wtg_day)
            st.write(f"Impacto Financeiro Anual: R${financial_impact[0]:,.2f}")
            st.write(f"Perda/Ganho Energético Anual: {energy_loss[0]:,.2f} MWh")
            fig_corr = px.scatter(x=df_selected[date_cols].iloc[0], y=financial_impact, title="Correlação Disponibilidade x Impacto Financeiro")
            st.plotly_chart(fig_corr)

        with tab4:
            st.subheader("Modelagem Probabilística")
            data = df_selected[date_cols].iloc[0].dropna()
            dist_name, params = fit_distribution(data)
            st.write(f"Melhor Distribuição Ajustada: {dist_name}")
            prob_below = stats.norm.cdf(contractual_availability, *params) if dist_name == 'norm' else stats.weibull_min.cdf(contractual_availability, *params)
            st.write(f"Probabilidade de Disponibilidade < Contratual: {prob_below:.2%}")

            # Histograma com curva ajustada
            fig_hist = px.histogram(data, nbins=20, histnorm='probability density')
            x = np.linspace(min(data), max(data), 100)
            if dist_name == 'norm':
                y = stats.norm.pdf(x, *params)
            elif dist_name == 'weibull_min':
                y = stats.weibull_min.pdf(x, *params)
            fig_hist.add_trace(go.Scatter(x=x, y=y, mode='lines', name=dist_name))
            st.plotly_chart(fig_hist)

        # Download de resultados
        csv = df_selected.to_csv(index=False)
        st.download_button("Baixar Resultados em CSV", csv, "resultados.csv", "text/csv")
else:
    st.info("Por favor, carregue uma planilha Excel para começar a análise.")
