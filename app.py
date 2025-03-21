# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

# Configurações iniciais do Streamlit
st.set_page_config(page_title="Análise de Parques Eólicos", layout="wide")

# Função para carregar os dados com cache para otimização de performance
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        # Converter colunas de datas (exceto a coluna 'Windfarm' e 'WTGs') para datetime
        date_cols = [col for col in df.columns if col not in ['Windfarm', 'WTGs']]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], format='%m/%d/%Y')
            except Exception:
                st.error(f"Erro ao converter a coluna {col} para data. Verifique o formato.")
        return df
    except Exception as e:
        st.error("Erro ao carregar o arquivo. Certifique-se de que o formato está correto.")
        return None

# Sidebar - Inputs de parâmetros contratuais
st.sidebar.header("Parâmetros Contratuais")
fator_multa = st.sidebar.number_input("Fator Multa", value=1.0, step=0.1)
disp_contratual = st.sidebar.number_input("Disponibilidade Contratual (%)", value=95.0, step=0.1)
fator_bonus = st.sidebar.number_input("Fator Bônus", value=1.0, step=0.1)
pmd = st.sidebar.number_input("PMD (R$/MWh)", value=100.0, step=1.0)
mwh_por_wtg = st.sidebar.number_input("MWh por WTG/dia", value=20.0, step=1.0)
periodos_mm = st.sidebar.number_input("Períodos para Média Móvel", min_value=1, value=3, step=1)

# Upload de arquivo
st.header("Upload dos Dados")
uploaded_file = st.file_uploader("Selecione a planilha Excel com os dados", type=["xlsx"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.success("Arquivo carregado com sucesso!")
        
        # Seleção do parque eólico (caso haja mais de um)
        parques = df['Windfarm'].unique()
        parque_selecionado = st.selectbox("Selecione o parque eólico", parques)
        df_parque = df[df['Windfarm'] == parque_selecionado].reset_index(drop=True)
        
        # Processamento dos dados
        # Supomos que as colunas de datas sejam as demais colunas
        date_cols = [col for col in df_parque.columns if col not in ['Windfarm', 'WTGs']]
        # Converter DataFrame para formato "long" para facilitar as análises temporais
        df_long = pd.melt(df_parque, id_vars=['Windfarm', 'WTGs'], 
                          value_vars=date_cols, var_name='Data', value_name='Disponibilidade')
        # Converter a coluna de datas para datetime, se necessário
        df_long['Data'] = pd.to_datetime(df_long['Data'], format='%m/%d/%Y', errors='coerce')
        df_long.dropna(subset=['Data'], inplace=True)
        df_long.sort_values('Data', inplace=True)
        
        # Cálculo de médias anuais (média aritmética das disponibilidades mensais)
        df_long['Ano'] = df_long['Data'].dt.year
        medias_anuais = df_long.groupby('Ano')['Disponibilidade'].mean().reset_index()
        
        # Cálculo de médias móveis
        medias_anuais['Média_Móvel'] = medias_anuais['Disponibilidade'].rolling(window=periodos_mm).mean()
        
        # Análise de distribuição dos dados mensais
        disponibilidades = df_long['Disponibilidade'].dropna()
        distribuições = {
            "Normal": stats.norm,
            "Log-Normal": stats.lognorm,
            "Beta": stats.beta,
            "Weibull": stats.exponweib  # ou stats.weibull_min, dependendo da modelagem desejada
        }
        melhor_ajuste = None
        melhor_pvalor = -np.inf
        ajuste_resultados = {}

        for nome, dist in distribuições.items():
            try:
                # Ajuste dos parâmetros à distribuição
                params = dist.fit(disponibilidades)
                # Teste de aderência (usando, por exemplo, o teste de Kolmogorov-Smirnov)
                ks_stat, p_value = stats.kstest(disponibilidades, nome.lower(), args=params)
                ajuste_resultados[nome] = {"params": params, "ks_stat": ks_stat, "p_value": p_value}
                if p_value > melhor_pvalor:
                    melhor_pvalor = p_value
                    melhor_ajuste = nome
            except Exception as e:
                ajuste_resultados[nome] = {"error": str(e)}
        
        # Aba de "Visão Anual"
        with st.tabs(["Visão Anual", "Análise Temporal", "Impacto Energético", "Modelagem Probabilística"])[0]:
            st.subheader("Métricas Anuais")
            st.dataframe(medias_anuais)
            
            # Gráfico de série temporal com média móvel e linha contratual
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=medias_anuais['Ano'], y=medias_anuais['Disponibilidade'],
                                      mode='lines+markers', name='Disponibilidade Anual'))
            fig1.add_trace(go.Scatter(x=medias_anuais['Ano'], y=medias_anuais['Média_Móvel'],
                                      mode='lines', name=f'Média Móvel ({periodos_mm} períodos)'))
            fig1.add_hline(y=disp_contratual, line_dash="dash", 
                           annotation_text="Disponibilidade Contratual", 
                           annotation_position="bottom right")
            st.plotly_chart(fig1, use_container_width=True)
            
        # Aba de "Análise Temporal"
        with st.tabs(["Visão Anual", "Análise Temporal", "Impacto Energético", "Modelagem Probabilística"])[1]:
            st.subheader("Análise Temporal e Decomposição Sazonal")
            # Decomposição sazonal - considerando uma série temporal diária ou mensal
            # Aqui agrupamos por data para média diária/mensal, conforme a granularidade dos dados
            ts = df_long.set_index('Data').resample('M').mean()['Disponibilidade']
            try:
                decomposed = seasonal_decompose(ts, model='additive', period=12)
                st.markdown("**Componentes da Decomposição Sazonal:**")
                st.line_chart(pd.DataFrame({
                    "Tendência": decomposed.trend,
                    "Sazonalidade": decomposed.seasonal,
                    "Resíduos": decomposed.resid
                }))
            except Exception as e:
                st.error("Erro na decomposição sazonal: " + str(e))
            
        # Aba de "Impacto Energético"
        with st.tabs(["Visão Anual", "Análise Temporal", "Impacto Energético", "Modelagem Probabilística"])[2]:
            st.subheader("Impacto Financeiro e Energético")
            # Cálculo de multas e bônus com base na fórmula contratual (exemplo ilustrativo)
            # Fórmula: Diferença entre Disponibilidade Real e Contratual * Fator Multa/Bônus
            medias_anuais['Diferença'] = medias_anuais['Disponibilidade'] - disp_contratual
            medias_anuais['Impacto'] = np.where(medias_anuais['Diferença'] < 0,
                                                 -medias_anuais['Diferença'] * fator_multa * pmd,
                                                 medias_anuais['Diferença'] * fator_bonus * pmd)
            # Projeção de perdas energéticas (MWh/ano)
            # Exemplo: perda energética = número de WTG * (diferença em % / 100) * MWh por WTG/dia * 365
            try:
                wtgs = df_parque['WTGs'].iloc[0]
            except:
                wtgs = 0
            medias_anuais['Perda_Energetica'] = wtgs * (np.abs(medias_anuais['Diferença']) / 100) * mwh_por_wtg * 365
            st.dataframe(medias_anuais[['Ano', 'Disponibilidade', 'Média_Móvel', 'Impacto', 'Perda_Energetica']])
            
            # Gráfico de correlação entre disponibilidade e perdas financeiras
            fig2 = px.scatter(medias_anuais, x='Disponibilidade', y='Impacto', trendline='ols',
                              title="Correlação entre Disponibilidade e Impacto Financeiro")
            st.plotly_chart(fig2, use_container_width=True)
            
        # Aba de "Modelagem Probabilística"
        with st.tabs(["Visão Anual", "Análise Temporal", "Impacto Energético", "Modelagem Probabilística"])[3]:
            st.subheader("Ajuste de Distribuições e Probabilidades")
            st.markdown("**Ajuste das distribuições aos dados mensais:**")
            if melhor_ajuste:
                st.write(f"A melhor distribuição ajustada: **{melhor_ajuste}** com p-valor: {melhor_pvalor:.4f}")
            else:
                st.write("Não foi possível determinar uma melhor distribuição.")
            st.write("Resultados dos ajustes:")
            for nome, res in ajuste_resultados.items():
                st.write(f"**{nome}**: {res}")
            
            # Cálculo de probabilidade de ficar abaixo da disponibilidade contratual
            prob_empirica = (disponibilidades < disp_contratual).mean()
            st.write(f"Probabilidade empírica de disponibilidade abaixo do contratual: {prob_empirica:.2%}")
            
            # Intervalos de confiança via bootstrap (95%)
            boot_samples = 1000
            boot_means = []
            for i in range(boot_samples):
                sample = disponibilidades.sample(frac=1, replace=True)
                boot_means.append(sample.mean())
            ci_lower = np.percentile(boot_means, 2.5)
            ci_upper = np.percentile(boot_means, 97.5)
            st.write(f"Intervalo de confiança (95%) para a média: [{ci_lower:.2f}, {ci_upper:.2f}]")
        
        # Botão para download dos resultados (CSV)
        csv = medias_anuais.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download dos Resultados em CSV",
            data=csv,
            file_name='resultados_medias_anuais.csv',
            mime='text/csv',
        )
