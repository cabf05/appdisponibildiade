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

# Função para carregar e tratar os dados, com cache para performance
@st.cache_data
def load_data(uploaded_file):
    try:
        # Lê o arquivo Excel usando openpyxl
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        # Verifica se as colunas mínimas existem
        if not set(['Windfarm', 'WTGs']).issubset(df.columns):
            st.error("O arquivo deve conter, no mínimo, as colunas 'Windfarm' e 'WTGs'.")
            return None
        # Identifica as colunas que são datas (todas que não sejam Windfarm e WTGs)
        date_cols = [col for col in df.columns if col not in ['Windfarm', 'WTGs']]
        for col in date_cols:
            try:
                # Converter os nomes das colunas para datas se estiverem em formato string (MM/DD/AAAA)
                # Se a conversão falhar, manter a coluna original para tentar converter os valores
                df[col] = pd.to_datetime(col, format='%m/%d/%Y', errors='coerce') if isinstance(col, str) else col
            except Exception as e:
                st.error(f"Erro ao converter o nome da coluna {col}: {e}")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

# Função para transformar dados de wide para long e tratar datas
def prepare_long_format(df):
    # Seleciona colunas de data (considerando que os nomes agora são objetos datetime se a conversão foi bem-sucedida)
    date_cols = [col for col in df.columns if col not in ['Windfarm', 'WTGs']]
    # Caso as colunas não tenham sido convertidas, assume os nomes originais e tenta converter os valores
    df_long = pd.melt(df, id_vars=['Windfarm', 'WTGs'], value_vars=date_cols, 
                      var_name='Data', value_name='Disponibilidade')
    # Converte a coluna 'Data' para datetime
    df_long['Data'] = pd.to_datetime(df_long['Data'], format='%m/%d/%Y', errors='coerce')
    df_long.dropna(subset=['Data'], inplace=True)
    df_long.sort_values('Data', inplace=True)
    df_long['Ano'] = df_long['Data'].dt.year
    return df_long

# Sidebar: parâmetros contratuais e ajustes de análises
st.sidebar.header("Parâmetros Contratuais e Configurações")
fator_multa = st.sidebar.number_input("Fator Multa", value=1.0, step=0.1)
disp_contratual = st.sidebar.number_input("Disponibilidade Contratual (%)", value=95.0, step=0.1)
fator_bonus = st.sidebar.number_input("Fator Bônus", value=1.0, step=0.1)
pmd = st.sidebar.number_input("PMD (R$/MWh)", value=100.0, step=1.0)
mwh_por_wtg = st.sidebar.number_input("MWh por WTG/dia", value=20.0, step=1.0)
periodos_mm = st.sidebar.number_input("Períodos para Média Móvel", min_value=1, value=3, step=1)
boot_samples = st.sidebar.number_input("Amostras para Bootstrap", min_value=100, value=1000, step=100)

# Upload do arquivo Excel
st.title("Sistema de Análise de Parques Eólicos")
st.markdown("Faça o upload da planilha Excel contendo as colunas **Windfarm**, **WTGs** e as datas no formato `MM/DD/AAAA`.")

uploaded_file = st.file_uploader("Selecione o arquivo Excel", type=["xlsx"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.success("Arquivo carregado com sucesso!")
        
        # Seleção do parque eólico (caso haja mais de um)
        parques = df['Windfarm'].unique()
        parque_selecionado = st.selectbox("Selecione o parque eólico", parques)
        df_parque = df[df['Windfarm'] == parque_selecionado].reset_index(drop=True)
        
        # Preparação dos dados em formato long
        df_long = prepare_long_format(df_parque)
        
        # Cálculo de médias anuais (média aritmética das disponibilidades mensais)
        medias_anuais = df_long.groupby('Ano')['Disponibilidade'].mean().reset_index()
        medias_anuais = medias_anuais.sort_values('Ano')
        # Cálculo da média móvel
        medias_anuais['Média_Móvel'] = medias_anuais['Disponibilidade'].rolling(window=periodos_mm).mean()
        
        # Análise de distribuição: melhor ajuste entre distribuições escolhidas
        disponibilidades = df_long['Disponibilidade'].dropna()
        distribuicoes = {
            "Normal": stats.norm,
            "Log-Normal": stats.lognorm,
            "Beta": stats.beta,
            "Weibull": stats.weibull_min  # Usando weibull_min para modelagem
        }
        melhor_ajuste = None
        melhor_pvalor = -np.inf
        ajuste_resultados = {}
        
        for nome, dist in distribuicoes.items():
            try:
                # Ajuste da distribuição
                params = dist.fit(disponibilidades)
                # Para o teste de Kolmogorov-Smirnov, usamos a função cdf da distribuição com os parâmetros ajustados.
                ks_stat, p_value = stats.kstest(disponibilidades, dist.name, args=params)
                ajuste_resultados[nome] = {"params": params, "ks_stat": ks_stat, "p_value": p_value}
                if p_value > melhor_pvalor:
                    melhor_pvalor = p_value
                    melhor_ajuste = nome
            except Exception as e:
                ajuste_resultados[nome] = {"error": str(e)}
        
        # Cálculo do impacto financeiro e energético por ano
        # Diferença entre disponibilidade real e contratual
        medias_anuais['Diferença (%)'] = medias_anuais['Disponibilidade'] - disp_contratual
        # Impacto financeiro: penalidade (multa) ou bônus, usando PMD (R$/MWh)
        medias_anuais['Impacto Financeiro (R$)'] = np.where(
            medias_anuais['Diferença (%)'] < 0,
            -medias_anuais['Diferença (%)'] * fator_multa * pmd,
            medias_anuais['Diferença (%)'] * fator_bonus * pmd
        )
        # Projeção de perdas energéticas (MWh/ano)
        try:
            wtgs = float(df_parque['WTGs'].iloc[0])
        except:
            wtgs = 0
        medias_anuais['Perda Energética (MWh/ano)'] = wtgs * (np.abs(medias_anuais['Diferença (%)']) / 100) * mwh_por_wtg * 365

        # Layout com abas para as diferentes análises
        tabs = st.tabs(["Visão Anual", "Análise Temporal", "Impacto Energético", "Modelagem Probabilística"])

        # Aba 1: Visão Anual
        with tabs[0]:
            st.header("Visão Anual")
            st.subheader("Métricas Anuais")
            st.dataframe(medias_anuais)
            
            st.subheader("Série Temporal")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=medias_anuais['Ano'], y=medias_anuais['Disponibilidade'],
                                      mode='lines+markers', name='Disponibilidade Anual'))
            fig1.add_trace(go.Scatter(x=medias_anuais['Ano'], y=medias_anuais['Média_Móvel'],
                                      mode='lines', name=f'Média Móvel ({periodos_mm} períodos)'))
            fig1.add_hline(y=disp_contratual, line_dash="dash",
                           annotation_text="Disponibilidade Contratual",
                           annotation_position="bottom right")
            fig1.update_layout(xaxis_title="Ano", yaxis_title="Disponibilidade (%)")
            st.plotly_chart(fig1, use_container_width=True)
        
        # Aba 2: Análise Temporal
        with tabs[1]:
            st.header("Análise Temporal e Decomposição Sazonal")
            # Agrupa os dados por mês (para a decomposição, usamos a média mensal)
            ts = df_long.set_index('Data').resample('M').mean()['Disponibilidade']
            if len(ts) < 24:
                st.warning("Dados insuficientes para decomposição sazonal robusta (mínimo recomendado: 24 meses).")
            else:
                try:
                    decomposed = seasonal_decompose(ts, model='additive', period=12)
                    df_decomp = pd.DataFrame({
                        "Tendência": decomposed.trend,
                        "Sazonalidade": decomposed.seasonal,
                        "Resíduos": decomposed.resid
                    })
                    st.subheader("Componentes da Decomposição")
                    st.line_chart(df_decomp)
                except Exception as e:
                    st.error(f"Erro na decomposição sazonal: {e}")
        
        # Aba 3: Impacto Energético
        with tabs[2]:
            st.header("Impacto Financeiro e Energético")
            st.subheader("Resultados Consolidados")
            st.dataframe(medias_anuais[['Ano', 'Disponibilidade', 'Média_Móvel', 
                                         'Diferença (%)', 'Impacto Financeiro (R$)', 
                                         'Perda Energética (MWh/ano)']])
            st.subheader("Correlação entre Disponibilidade e Impacto Financeiro")
            fig2 = px.scatter(medias_anuais, x='Disponibilidade', y='Impacto Financeiro (R$)', trendline='ols',
                              title="Correlação entre Disponibilidade e Impacto Financeiro")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Aba 4: Modelagem Probabilística
        with tabs[3]:
            st.header("Modelagem Probabilística")
            st.subheader("Ajuste de Distribuições")
            if melhor_ajuste:
                st.write(f"A melhor distribuição ajustada: **{melhor_ajuste}** (p-valor: {melhor_pvalor:.4f})")
            else:
                st.write("Não foi possível determinar uma distribuição ajustada de forma confiável.")
            st.markdown("**Detalhes dos Ajustes:**")
            for nome, res in ajuste_resultados.items():
                st.write(f"**{nome}**: {res}")
            
            st.subheader("Probabilidade de Disponibilidade Abaixo do Contratual")
            prob_empirica = (disponibilidades < disp_contratual).mean()
            st.write(f"Probabilidade empírica: **{prob_empirica:.2%}**")
            
            st.subheader("Intervalo de Confiança (95%) via Bootstrap")
            boot_means = []
            for i in range(int(boot_samples)):
                sample = disponibilidades.sample(frac=1, replace=True)
                boot_means.append(sample.mean())
            ci_lower = np.percentile(boot_means, 2.5)
            ci_upper = np.percentile(boot_means, 97.5)
            st.write(f"Intervalo de confiança para a média: **[{ci_lower:.2f}, {ci_upper:.2f}]**")
        
        # Botão de download dos resultados consolidados (CSV)
        csv = medias_anuais.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download dos Resultados (CSV)",
            data=csv,
            file_name='resultados_medias_anuais.csv',
            mime='text/csv'
        )
else:
    st.info("Aguardando upload do arquivo Excel...")
