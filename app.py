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

# Função para converter colunas corretamente
def preprocess_data(df):
    df = df.copy()
    
    # Convertendo datas e valores numéricos
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Ignora valores inválidos
    
    return df

# Função para calcular métricas principais
def calcular_metricas(df, disponibilidade_contratual, fator_multa, fator_bonus, pmd, mwh_por_wtg):
    metricas = []

    for parque in df['Windfarm'].unique():
        df_parque = df[df['Windfarm'] == parque].iloc[:, 2:]
        
        # Ignorar NaN e calcular métricas apenas com valores válidos
        df_parque_clean = df_parque.dropna(axis=1, how='all')
        
        if df_parque_clean.empty:
            continue  # Pula parques sem dados válidos
        
        media_anual = df_parque_clean.mean().mean()
        desvio_padrao = df_parque_clean.mean().std()
        perda_mwh = (disponibilidade_contratual - media_anual) * mwh_por_wtg * len(df_parque_clean)

        multa = abs(perda_mwh) * pmd * fator_multa if media_anual < disponibilidade_contratual else 0
        bonus = abs(perda_mwh) * pmd * fator_bonus if media_anual >= disponibilidade_contratual else 0

        metricas.append({
            "Parque": parque,
            "Disponibilidade Média (%)": round(media_anual, 2),
            "Desvio Padrão": round(desvio_padrao, 2) if not np.isnan(desvio_padrao) else 0,
            "Perda Energética (MWh)": round(perda_mwh, 2),
            "Multa (R$)": round(multa, 2),
            "Bônus (R$)": round(bonus, 2),
        })

    return pd.DataFrame(metricas)

# Função para análise estatística
def analise_probabilistica(df, disponibilidade_contratual):
    df_prob = df.copy()
    if "Desvio Padrão" in df_prob.columns and df_prob["Desvio Padrão"].mean() > 0:
        df_prob["Probabilidade Abaixo do Contrato (%)"] = df_prob["Disponibilidade Média (%)"].apply(
            lambda x: round(stats.norm.cdf(disponibilidade_contratual, loc=x, scale=df_prob["Desvio Padrão"].mean()) * 100, 2)
        )
    else:
        df_prob["Probabilidade Abaixo do Contrato (%)"] = 0

    return df_prob

# Interface do Streamlit
st.title("📊 Análise de Disponibilidade de Parques Eólicos")

# Sidebar para upload e entrada de parâmetros
st.sidebar.header("Configurações")

uploaded_file = st.sidebar.file_uploader("📂 Faça o upload da planilha Excel", type=["xlsx"])
disponibilidade_contratual = st.sidebar.slider("Disponibilidade Contratual (%)", 80, 100, 95)
fator_multa = st.sidebar.number_input("Fator de Multa", min_value=0.0, value=1.5)
fator_bonus = st.sidebar.number_input("Fator de Bônus", min_value=0.0, value=1.2)
pmd = st.sidebar.number_input("PMD (R$/MWh)", min_value=0.0, value=300.0)
mwh_por_wtg = st.sidebar.number_input("MWh por WTG/dia", min_value=0.0, value=2.0)
periodos_mm = st.sidebar.slider("Períodos para Média Móvel", 1, 12, 3)

# Processamento após upload do arquivo
if uploaded_file:
    df = load_data(uploaded_file)
    df = preprocess_data(df)

    if df is not None:
        st.write("### 🔍 Pré-visualização dos Dados")
        st.dataframe(df.head())

        # Cálculo de métricas
        df_metricas = calcular_metricas(df, disponibilidade_contratual, fator_multa, fator_bonus, pmd, mwh_por_wtg)
        df_metricas = analise_probabilistica(df_metricas, disponibilidade_contratual)

        # Exibir métricas
        st.write("### 📈 Métricas dos Parques Eólicos")
        st.dataframe(df_metricas)

        # Gráfico de comparação entre parques
        fig = px.bar(df_metricas, x="Parque", y="Disponibilidade Média (%)", color="Parque", title="Disponibilidade Média por Parque")
        st.plotly_chart(fig, use_container_width=True)

        # Série temporal
        df_melted = df.melt(id_vars=["Windfarm"], var_name="Data", value_name="Disponibilidade")
        df_melted["Data"] = pd.to_datetime(df_melted["Data"], errors='coerce')
        df_melted = df_melted.dropna()

        if not df_melted.empty:
            st.write("### 📅 Análise Temporal")
            parque_selecionado = st.selectbox("Selecione um parque para análise:", df_melted["Windfarm"].unique())

            df_filtrado = df_melted[df_melted["Windfarm"] == parque_selecionado].groupby("Data", as_index=True)["Disponibilidade"].mean()
            df_filtrado = df_filtrado.dropna()
            df_filtrado = df_filtrado.to_frame()  # Converter para DataFrame se necessário
            
            df_filtrado["Média Móvel"] = df_filtrado["Disponibilidade"].rolling(window=periodos_mm, min_periods=1).mean()

            fig = px.line(df_filtrado, x=df_filtrado.index, y=["Disponibilidade", "Média Móvel"], 
                          title=f"Evolução da Disponibilidade - {parque_selecionado}")
            st.plotly_chart(fig, use_container_width=True)

            # Decomposição sazonal da série temporal
            if len(df_filtrado) >= periodos_mm * 2:  # Evitar erro se a série for muito curta
                st.write("### 🔬 Análise de Curva Sazonal")
                decomposed = seasonal_decompose(df_filtrado["Disponibilidade"], model="additive", period=periodos_mm)
                
                st.line_chart(decomposed.trend.rename("Tendência"))
                st.line_chart(decomposed.seasonal.rename("Sazonalidade"))
                st.line_chart(decomposed.resid.rename("Resíduos"))

        # Botão de download dos resultados
        csv_data = df_metricas.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar Resultados em CSV", data=csv_data, file_name="analise_parques_eolicos.csv", mime="text/csv")

else:
    st.warning("📌 Por favor, faça o upload de um arquivo Excel para iniciar a análise.")
