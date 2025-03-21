import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Configuração da interface
st.title("Análise de Disponibilidade de Parques Eólicos")

# Barra lateral para inputs
st.sidebar.header("Parâmetros")
fator_multa = st.sidebar.number_input("Fator Multa", min_value=0.0, value=1.0, step=0.1)
disp_contratual = st.sidebar.number_input("Disponibilidade Contratual (%)", min_value=0.0, max_value=100.0, value=95.0, step=0.1)
fator_bonus = st.sidebar.number_input("Fator Bônus", min_value=0.0, value=0.5, step=0.1)
pmd = st.sidebar.number_input("PMD", min_value=0.0, value=100.0, step=1.0)
mwh_por_wtg = st.sidebar.number_input("MWh por WTG", min_value=0.0, value=500.0, step=10.0)

# Upload da planilha
uploaded_file = st.file_uploader("Carregue sua planilha Excel", type=["xlsx"])
if uploaded_file is not None:
    # Ler a planilha
    df = pd.read_excel(uploaded_file)
    
    # Converter colunas de data para formato datetime
    date_cols = [col for col in df.columns if col not in ['Windfarm', 'WTGs']]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format='%m/%d/%Y', errors='coerce')
    
    # Organizar dados
    windfarms = df['Windfarm'].unique()
    st.subheader("Resultados por Parque Eólico")
    
    for windfarm in windfarms:
        st.write(f"### {windfarm}")
        wf_data = df[df['Windfarm'] == windfarm].iloc[0]
        wtgs = wf_data['WTGs']
        
        # Extrair disponibilidades mensais
        disp_mensal = wf_data[date_cols].values
        anos = pd.to_datetime(date_cols).year.unique()
        
        # Calcular médias anuais
        medias_anuais = {}
        for ano in anos:
            cols_ano = [col for col in date_cols if pd.to_datetime(col).year == ano]
            medias_anuais[ano] = np.mean(wf_data[cols_ano])
        
        # Exibir médias anuais
        st.write("#### Médias Anuais de Disponibilidade (%)")
        st.write(pd.Series(medias_anuais))
        
        # Estatísticas adicionais
        disp_anuais = list(medias_anuais.values())
        stats_dict = {
            "Mediana (%)": np.median(disp_anuais),
            "Desvio Padrão (%)": np.std(disp_anuais),
            "Mínimo (%)": np.min(disp_anuais),
            "Máximo (%)": np.max(disp_anuais)
        }
        st.write("#### Estatísticas das Disponibilidades Anuais")
        st.write(pd.Series(stats_dict))
        
        # Ajuste de distribuição para disponibilidades mensais
        distribs = {'Normal': stats.norm, 'Log-Normal': stats.lognorm, 'Weibull': stats.weibull_min}
        best_dist = None
        best_pvalue = 0
        for name, dist in distribs.items():
            if name == 'Log-Normal':
                params = dist.fit(disp_mensal, floc=0)
            elif name == 'Weibull':
                params = dist.fit(disp_mensal, floc=0)
            else:
                params = dist.fit(disp_mensal)
            ks_stat, p_value = stats.kstest(disp_mensal, name.lower(), args=params)
            if p_value > best_pvalue:
                best_pvalue = p_value
                best_dist = (name, dist, params)
        
        # Calcular probabilidades mensais
        if best_pvalue > 0.05:  # Ajuste aceitável
            dist_name, dist, params = best_dist
            prob_mensal = dist.cdf(disp_contratual, *params)
            confianca = "Alta (p-valor > 0.05)"
        else:  # Método alternativo (não paramétrico)
            prob_mensal = np.mean(disp_mensal < disp_contratual)
            confianca = "Média (método não paramétrico)"
        
        st.write(f"#### Probabilidade Mensal de Disponibilidade < {disp_contratual}%")
        st.write(f"Probabilidade: {prob_mensal:.2%} (Confiança: {confianca})")
        
        # Probabilidade anual
        if len(disp_anuais) > 5:  # Necessário número mínimo de anos
            params_anual = stats.norm.fit(disp_anuais)
            prob_anual = stats.norm.cdf(disp_contratual, *params_anual)
            confianca_anual = "Alta (assumindo normalidade)"
        else:
            prob_anual = np.mean(np.array(disp_anuais) < disp_contratual)
            confianca_anual = "Média (método não paramétrico)"
        
        st.write(f"#### Probabilidade Anual de Disponibilidade < {disp_contratual}%")
        st.write(f"Probabilidade: {prob_anual:.2%} (Confiança: {confianca_anual})")
        
        # Calcular multas e bônus
        st.write("#### Multas e Bônus por Ano")
        resultados = {}
        for ano, disp in medias_anuais.items():
            if disp < disp_contratual:
                multa = fator_multa * pmd * mwh_por_wtg * wtgs * (disp_contratual / disp - 1)
                resultados[ano] = f"Multa: R${multa:,.2f}"
            elif disp > disp_contratual:
                bonus = fator_bonus * pmd * mwh_por_wtg * wtgs * (disp / disp_contratual - 1)
                resultados[ano] = f"Bônus: R${bonus:,.2f}"
            else:
                resultados[ano] = "Neutro"
        st.write(pd.Series(resultados))
        
        # Gráfico das disponibilidades anuais
        fig, ax = plt.subplots()
        ax.plot(list(medias_anuais.keys()), list(medias_anuais.values()), marker='o')
        ax.axhline(disp_contratual, color='r', linestyle='--', label='Disp. Contratual')
        ax.set_title(f"Disponibilidade Anual - {windfarm}")
        ax.set_xlabel("Ano")
        ax.set_ylabel("Disponibilidade (%)")
        ax.legend()
        st.pyplot(fig)

else:
    st.write("Por favor, faça o upload da planilha para iniciar a análise.")
