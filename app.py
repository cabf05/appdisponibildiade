import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from io import BytesIO
import calendar

# Configuração da página
st.set_page_config(page_title="Análise Avançada de Parques Eólicos", layout="wide")

# Sidebar: Inputs e Upload
with st.sidebar:
    st.header("Parâmetros Contratuais")
    fator_multa = st.number_input("Fator Multa", min_value=0.0, value=1.0, step=0.1)
    disp_contratual = st.number_input("Disponibilidade Contratual (%)", min_value=0.0, 
                                     max_value=100.0, value=95.0, step=0.1)
    fator_bonus = st.number_input("Fator Bônus", min_value=0.0, value=0.5, step=0.1)
    pmd = st.number_input("PMD (R$/MWh)", min_value=0.0, value=150.0)
    mwh_por_wtg = st.number_input("MWh por WTG/dia", min_value=0.0, value=3.5)
    
    st.header("Configurações de Análise")
    window_size = st.number_input("Períodos para Média Móvel", min_value=1, max_value=36, value=12)
    n_bootstrap = st.number_input("Amostras Bootstrap", min_value=100, max_value=10000, value=1000)
    
    uploaded_file = st.file_uploader("Upload da Planilha (XLSX)", type=["xlsx"])

# Processamento de Dados
@st.cache_data
def process_data(uploaded_file):
    if uploaded_file is None:
        return None
    
    df = pd.read_excel(uploaded_file)
    
    # Verificar estrutura
    required_cols = ["Windfarm", "WTGs"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Colunas obrigatórias não encontradas")
    
    # Processar datas
    date_columns = [col for col in df.columns if '/' in col and len(col.split('/')) == 3]
    df_dates = df.melt(id_vars=required_cols, value_vars=date_columns, 
                      var_name="Data", value_name="Disponibilidade")
    
    try:
        df_dates['Data'] = pd.to_datetime(df_dates['Data'], format='%m/%d/%Y')
    except:
        st.error("Formato de data inválido. Use MM/DD/YYYY")
        return None
    
    df_dates['Ano'] = df_dates['Data'].dt.year
    df_dates['Mês'] = df_dates['Data'].dt.month
    df_dates['Mês_Ano'] = df_dates['Data'].dt.to_period('M')
    
    # Calcular métricas anuais
    df_anual = df_dates.groupby(['Windfarm', 'Ano']).agg(
        Disponibilidade_Anual=('Disponibilidade', 'mean'),
        WTGs=('WTGs', 'first'),
        Meses_Coletados=('Disponibilidade', 'count')
    ).reset_index()
    
    # Calcular métricas móveis
    df_dates['Media_Movel'] = df_dates.groupby('Windfarm')['Disponibilidade'] \
                                     .transform(lambda x: x.rolling(window=window_size).mean())
    
    return df_dates, df_anual

# Funções de Análise
def fit_best_distribution(data):
    dist_names = ['norm', 'lognorm', 'beta', 'weibull_min']
    results = []
    
    for dist_name in dist_names:
        dist = getattr(stats, dist_name)
        try:
            params = dist.fit(data)
            aic = dist.aic(params, data)
            results.append((dist_name, aic))
        except:
            continue
    
    if not results:
        return None
    results.sort(key=lambda x: x[1])
    return results[0][0]

def bootstrap_probability(data, threshold, n_samples):
    probs = []
    for _ in range(n_samples):
        sample = np.random.choice(data, size=len(data), replace=True)
        probs.append(np.mean(sample < threshold))
    return np.percentile(probs, [2.5, 97.5])

# Interface Principal
if uploaded_file:
    try:
        df_dates, df_anual = process_data(uploaded_file)
        windfarms = df_dates['Windfarm'].unique()
        selected_windfarm = st.selectbox("Selecione o Parque Eólico", windfarms)
        
        df_windfarm = df_dates[df_dates['Windfarm'] == selected_windfarm]
        df_anual_windfarm = df_anual[df_anual['Windfarm'] == selected_windfarm]
        
        # Abas de Análise
        tab1, tab2, tab3, tab4 = st.tabs([
            "Visão Anual", 
            "Análise Temporal", 
            "Impacto Energético",
            "Modelagem Probabilística"
        ])
        
        with tab1:
            st.subheader("Desempenho Anual")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Média Histórica", f"{df_anual_windfarm['Disponibilidade_Anual'].mean():.1f}%")
                st.dataframe(df_anual_windfarm.set_index('Ano'))
                
            with col2:
                fig = px.bar(df_anual_windfarm, x='Ano', y='Disponibilidade_Anual',
                            title="Disponibilidade Anual vs Contratual",
                            labels={'Disponibilidade_Anual': 'Disponibilidade (%)'})
                fig.add_hline(y=disp_contratual, line_dash="dot", 
                             annotation_text=f"Contratual: {disp_contratual}%")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Análise Temporal Detalhada")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            df_windfarm.set_index('Data')['Disponibilidade'].plot(ax=ax, label='Mensal')
            df_windfarm.set_index('Data')['Media_Movel'].plot(ax=ax, label=f'Média Móvel ({window_size} meses)')
            ax.axhline(disp_contratual, color='r', linestyle='--', label='Contratual')
            ax.set_title("Série Temporal de Disponibilidade")
            ax.legend()
            st.pyplot(fig)
            
            st.subheader("Decomposição Sazonal")
            decomposition = seasonal_decompose(df_windfarm.set_index('Data')['Disponibilidade'], model='additive')
            fig = decomposition.plot()
            st.pyplot(fig)
        
        with tab3:
            st.subheader("Impacto Financeiro")
            df_anual_windfarm['Perda_Energia'] = ((disp_contratual - df_anual_windfarm['Disponibilidade_Anual'])/100 * 
                                                df_anual_windfarm['WTGs'] * mwh_por_wtg * 365)
            df_anual_windfarm['Perda_Monetaria'] = df_anual_windfarm['Perda_Energia'] * pmd
            
            st.dataframe(df_anual_windfarm[['Ano', 'Perda_Energia', 'Perda_Monetaria']].set_index('Ano'))
            
            fig = px.scatter(df_anual_windfarm, x='Disponibilidade_Anual', y='Perda_Monetaria',
                            trendline="ols", title="Correlação Disponibilidade vs Perdas")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Modelagem Probabilística")
            
            best_dist = fit_best_distribution(df_windfarm['Disponibilidade'])
            prob_mes = stats.percentileofscore(df_windfarm['Disponibilidade'], disp_contratual)/100
            
            if best_dist:
                dist = getattr(stats, best_dist)
                params = dist.fit(df_windfarm['Disponibilidade'])
                prob_mes_dist = dist.cdf(disp_contratual, *params)
                ci = bootstrap_probability(df_windfarm['Disponibilidade'], disp_contratual, n_bootstrap)
                
                st.markdown(f"""
                **Melhor Distribuição Ajustada:** `{best_dist}`  
                **Probabilidade Mensal (Distribuição):** {prob_mes_dist:.1%}  
                **Probabilidade Mensal (Empírica):** {prob_mes:.1%}  
                **Intervalo de Confiança 95% (Empírico):** [{ci[0]:.1%} - {ci[1]:.1%}]
                """)
            else:
                st.warning("Nenhuma distribuição adequada encontrada. Usando método empírico.")
                st.write(f"Probabilidade Mensal: {prob_mes:.1%}")
            
            st.subheader("Análise de Risco Anual")
            prob_ano_emp = np.mean(df_anual_windfarm['Disponibilidade_Anual'] < disp_contratual)
            st.write(f"Probabilidade Anual (Histórica): {prob_ano_emp:.1%}")
            
    except Exception as e:
        st.error(f"Erro no processamento: {str(e)}")

else:
    st.info("Faça upload da planilha para iniciar a análise.")
