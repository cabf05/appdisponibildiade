import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
from statsmodels.tsa.seasonal import seasonal_decompose

# Configuração inicial
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
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Ignora valores inválidos
    return df

# Função para calcular métricas principais
def calcular_metricas(df, disponibilidade_contratual, fator_multa, fator_bonus, pmd, mwh_por_wtg, num_wtg):
    metricas = []
    total_multa_projeto = 0
    total_bonus_projeto = 0

    for parque in df['Windfarm'].unique():
        df_parque = df[df['Windfarm'] == parque].iloc[:, 2:]
        df_parque_clean = df_parque.dropna(axis=1, how='all')
        
        if df_parque_clean.empty:
            continue
        
        # Disponibilidade média anual do parque
        media_anual = df_parque_clean.mean().mean()
        
        # Cálculo do MWh anual por parque
        mwh_anual_por_wtg = mwh_por_wtg * 365  # 365 dias no ano
        mwh_anual_parque = mwh_anual_por_wtg * num_wtg
        
        # Cálculo de multa ou bônus
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
            "Disponibilidade Média Anual (%)": round(media_anual, 2),
            "MWh Anual": round(mwh_anual_parque, 2),
            "Multa (R$)": round(multa, 2),
            "Bônus (R$)": round(bonus, 2),
        })

    df_metricas = pd.DataFrame(metricas)
    df_metricas["Total Multa Projeto (R$)"] = round(total_multa_projeto, 2)
    df_metricas["Total Bônus Projeto (R$)"] = round(total_bonus_projeto, 2)
    
    return df_metricas

# Função para calcular a probabilidade de atingir um valor de multa
def calcular_prob_multa(df_metricas, valor_multa, disponibilidade_contratual, fator_multa, pmd, mwh_anual_parque):
    media_disponibilidade = df_metricas["Disponibilidade Média Anual (%)"].mean()
    desvio_padrao_disponibilidade = df_metricas["Disponibilidade Média Anual (%)"].std()
    
    def multa(disponibilidade):
        if disponibilidade < disponibilidade_contratual:
            return fator_multa * pmd * mwh_anual_parque * ((disponibilidade_contratual / disponibilidade) - 1)
        return 0
    
    # Simulação para estimar a probabilidade
    num_simulacoes = 10000
    disponibilidades_simuladas = np.random.normal(media_disponibilidade, desvio_padrao_disponibilidade, num_simulacoes)
    multas_simuladas = [multa(d) for d in disponibilidades_simuladas if d > 0]
    prob = np.mean([m > valor_multa for m in multas_simuladas if m is not None])
    
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
mwh_por_wtg = st.sidebar.number_input("MWh por WTG/dia", min_value=0.0, value=2.0)
num_wtg = st.sidebar.number_input("Número de WTGs por Parque", min_value=1, value=10)
valor_multa_desejado = st.sidebar.number_input("Valor de Multa para Probabilidade (R$)", min_value=0.0, value=100000.0)
periodos_mm = st.sidebar.slider("Períodos para Média Móvel", 1, 12, 3)

# Processamento após upload
if uploaded_file:
    df = load_data(uploaded_file)
    df = preprocess_data(df)

    if df is not None:
        st.write("### 🔍 Pré-visualização dos Dados")
        st.dataframe(df.head())

        # Cálculo de métricas
        df_metricas = calcular_metricas(df, disponibilidade_contratual, fator_multa, fator_bonus, pmd, mwh_por_wtg, num_wtg)

        # Exibir métricas
        st.write("### 📈 Métricas dos Parques Eólicos")
        st.dataframe(df_metricas)

        # Total do projeto
        total_multa = df_metricas["Total Multa Projeto (R$)"].iloc[0]
        total_bonus = df_metricas["Total Bônus Projeto (R$)"].iloc[0]
        st.write(f"#### Total Anual do Projeto")
        st.write(f"- **Multa Total**: R$ {total_multa:,.2f}")
        st.write(f"- **Bônus Total**: R$ {total_bonus:,.2f}")

        # Gráfico de disponibilidade por parque
        fig1 = px.bar(df_metricas, x="Parque", y="Disponibilidade Média Anual (%)", color="Parque",
                      title="Disponibilidade Média Anual por Parque",
                      labels={"Disponibilidade Média Anual (%)": "Disponibilidade (%)"})
        st.plotly_chart(fig1, use_container_width=True)
        st.write("*Gráfico mostrando a disponibilidade média anual de cada parque eólico.*")

        # Gráfico de multas e bônus por parque
        fig2 = px.bar(df_metricas, x="Parque", y=["Multa (R$)", "Bônus (R$)"], barmode="group",
                      title="Multas e Bônus Anuais por Parque",
                      labels={"value": "Valor (R$)", "variable": "Tipo"})
        st.plotly_chart(fig2, use_container_width=True)
        st.write("*Gráfico comparando as multas e bônus anuais calculados para cada parque.*")

        # Probabilidade de atingir um valor de multa
        mwh_anual_parque = mwh_por_wtg * 365 * num_wtg
        prob_multa = calcular_prob_multa(df_metricas, valor_multa_desejado, disponibilidade_contratual, fator_multa, pmd, mwh_anual_parque)
        st.write(f"### Probabilidade de Multa Superior a R$ {valor_multa_desejado:,.2f}")
        st.write(f"Probabilidade: **{prob_multa * 100:.2f}%**")
        st.write("*Esta probabilidade é calculada simulando a distribuição das disponibilidades médias anuais (assumidas como normais) e verificando em quantos casos a multa total ultrapassa o valor especificado.*")

        # Série temporal
        df_melted = df.melt(id_vars=["Windfarm"], var_name="Data", value_name="Disponibilidade")
        df_melted["Data"] = pd.to_datetime(df_melted["Data"], errors='coerce')
        df_melted = df_melted.dropna()

        if not df_melted.empty:
            st.write("### 📅 Análise Temporal")
            parque_selecionado = st.selectbox("Selecione um parque para análise:", df_melted["Windfarm"].unique())
            df_filtrado = df_melted[df_melted["Windfarm"] == parque_selecionado].groupby("Data")["Disponibilidade"].mean().dropna()
            df_filtrado = df_filtrado.to_frame()
            df_filtrado["Média Móvel"] = df_filtrado["Disponibilidade"].rolling(window=periodos_mm, min_periods=1).mean()

            fig3 = px.line(df_filtrado, x=df_filtrado.index, y=["Disponibilidade", "Média Móvel"],
                           title=f"Evolução da Disponibilidade - {parque_selecionado}",
                           labels={"value": "Disponibilidade (%)", "variable": "Métrica"})
            st.plotly_chart(fig3, use_container_width=True)
            st.write("*Gráfico da disponibilidade diária e sua média móvel ao longo do tempo para o parque selecionado.*")

            # Decomposição sazonal
            if len(df_filtrado) >= periodos_mm * 2:
                st.write("### 🔬 Análise de Curva Sazonal")
                decomposed = seasonal_decompose(df_filtrado["Disponibilidade"], model="additive", period=periodos_mm)
                st.line_chart(decomposed.trend.rename("Tendência"))
                st.write("*Tendência: Componente de longo prazo da disponibilidade.*")
                st.line_chart(decomposed.seasonal.rename("Sazonalidade"))
                st.write("*Sazonalidade: Padrões recorrentes na disponibilidade.*")
                st.line_chart(decomposed.resid.rename("Resíduos"))
                st.write("*Resíduos: Variações aleatórias após remover tendência e sazonalidade.*")

        # Download dos resultados
        csv_data = df_metricas.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar Resultados em CSV", data=csv_data, file_name="analise_parques_eolicos.csv", mime="text/csv")

else:
    st.warning("📌 Por favor, faça o upload de um arquivo Excel para iniciar a análise.")
