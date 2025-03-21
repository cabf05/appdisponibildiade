import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm, lognorm, skew, kurtosis, probplot

# Configuração da página
st.set_page_config(page_title="Análise Avançada de Parques Eólicos", layout="wide")

# Funções auxiliares
def analisar_distribuicao(dados):
    """Analisa dados e sugere distribuição mais adequada"""
    if len(dados) < 2:
        return "Normal", 0, 0
    
    skewness = skew(dados)
    curtose = kurtosis(dados)
    
    if abs(skewness) < 0.5 and 2.5 < curtose < 3.5:
        return "Normal", skewness, curtose
    elif skewness >= 0.5:
        return "Lognormal", skewness, curtose
    else:
        return "Beta", skewness, curtose

def plot_qq(dados, distribuicao, ax):
    """Gera Q-Q plot para análise de distribuição"""
    if distribuicao == "Normal":
        probplot(dados, dist="norm", plot=ax)
    elif distribuicao == "Lognormal":
        probplot(np.log(dados[dados > 0]), dist="norm", plot=ax)
    elif distribuicao == "Beta":
        dados_scaled = dados/100
        probplot(dados_scaled, dist="beta", sparams=(1,1), plot=ax)
    ax.set_title(f'Q-Q Plot vs {distribuicao}')

def simular_disponibilidade(media, std, distribuição, dados_historicos=None):
    """Simula valor de acordo com a distribuição escolhida"""
    try:
        if distribuição == "Normal":
            sim = np.random.normal(media, std)
        elif distribuição == "Beta":
            media /= 100
            std /= 100
            var = std**2
            alpha = ((1 - media) / var - 1/media) * media**2
            beta_param = alpha * (1/media - 1)
            sim = beta.rvs(alpha, beta_param) * 100
        elif distribuição == "Lognormal":
            sigma = np.sqrt(np.log(1 + (std**2)/(media**2)))
            mu = np.log(media) - 0.5*sigma**2
            sim = lognorm.rvs(sigma, scale=np.exp(mu))
        elif distribuição == "Bootstrap":
            sim = np.random.choice(dados_historicos, size=1)[0]
        return np.clip(sim, 0, 100)
    except:
        return media  # Fallback para média histórica

def processar_dados(arquivo):
    """Processa o arquivo Excel carregado"""
    df = pd.read_excel(arquivo, sheet_name="Sheet1")
    id_vars = ["Windfarm", "WTGs"]
    value_vars = df.columns[2:]
    df_melted = df.melt(
        id_vars=id_vars, 
        value_vars=value_vars, 
        var_name="Data", 
        value_name="Disponibilidade"
    )
    df_melted["Data"] = pd.to_datetime(df_melted["Data"])
    df_melted["Ano"] = df_melted["Data"].dt.year
    df_melted["Mês"] = df_melted["Data"].dt.month
    return df_melted

# Interface principal
def main():
    st.title("Sistema Completo de Análise de Parques Eólicos")
    
    # Upload do arquivo
    arquivo = st.file_uploader("Carregue a planilha de disponibilidade", type="xlsx")
    if not arquivo:
        st.info("Por favor, carregue um arquivo Excel no formato especificado.")
        return

    df = processar_dados(arquivo)
    
    # Tabela 1: WTGs por Parque
    st.header("1. Número de WTGs por Parque")
    tabela_wtgs = df.groupby("Windfarm")["WTGs"].first().reset_index()
    total_wtgs = tabela_wtgs["WTGs"].sum()
    st.markdown(f"**Total Geral de WTGs:** {total_wtgs}")
    st.dataframe(tabela_wtgs, use_container_width=True)

    # Tabela 2: Disponibilidade Mensal
    st.header("2. Disponibilidade por Ano e Mês")
    tabela_mensal = df.pivot_table(
        index=["Windfarm", "Ano"],
        columns="Mês",
        values="Disponibilidade",
        aggfunc="mean"
    ).reset_index().round(2)
    tabela_mensal.columns = [f"Mês {col}" if isinstance(col, int) else col for col in tabela_mensal.columns]
    st.dataframe(tabela_mensal, use_container_width=True)

    # Tabela 3: Média Anual
    st.header("3. Disponibilidade Média Anual")
    tabela_media_anual = df.groupby(["Windfarm", "Ano"])["Disponibilidade"].mean().reset_index()
    st.dataframe(tabela_media_anual, use_container_width=True)

    # Tabela 4: Desvio Padrão Anual
    st.header("4. Desvio Padrão Anual")
    tabela_std_anual = df.groupby(["Windfarm", "Ano"])["Disponibilidade"].std().reset_index()
    st.dataframe(tabela_std_anual.fillna(0), use_container_width=True)

    # Tabela 5: Cálculo de Penalidade
    st.header("5. Cálculo de Penalidade")
    contrato = st.number_input("Disponibilidade de Contrato (%)", 0.0, 100.0, 95.0, key="contrato")
    merged = pd.merge(tabela_media_anual, tabela_wtgs, on="Windfarm")
    merged["Tabela5"] = (contrato / merged["Disponibilidade"] - 1).round(4).apply(lambda x: max(x, 0))
    st.dataframe(merged[["Windfarm", "Ano", "Tabela5"]], use_container_width=True)

    # Tabela 6: Cálculo da Multa
    st.header("6. Cálculo da Multa")
    fator_multa = st.number_input("Fator de Multa (R$ por WTG)", 0.0, 100000.0, 1000.0, key="fator_multa")
    merged["Tabela6"] = merged["Tabela5"] * fator_multa * merged["WTGs"]
    st.dataframe(merged[["Windfarm", "Ano", "Tabela6"]], use_container_width=True)

    # Tabela 7: Soma Total de Multas
    st.header("7. Soma Total de Multas por Ano")
    soma_multa_ano = merged.groupby("Ano")["Tabela6"].sum().reset_index()
    st.dataframe(soma_multa_ano, use_container_width=True)

    # Seção de Simulação Avançada
    st.header("8. Simulação Estatística Avançada")
    
    # Análise de Distribuição
    st.subheader("Análise de Distribuição")
    parque_analise = st.selectbox("Selecione o Parque para Análise", df["Windfarm"].unique())
    dados_historicos = df[df["Windfarm"] == parque_analise]["Disponibilidade"]
    
    dist_sugerida, skew_val, kurt_val = analisar_distribuicao(dados_historicos)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Estatísticas Descritivas:**
        - Média: {dados_historicos.mean():.2f}%
        - Desvio Padrão: {dados_historicos.std():.2f}%
        - Assimetria: {skew_val:.2f}
        - Curtose: {kurt_val:.2f}
        """)
    
    with col2:
        st.markdown(f"""
        **Recomendação do Sistema:**
        - Distribuição Sugerida: **{dist_sugerida}**
        - Justificativa: {"Dados simétricos" if dist_sugerida == "Normal" 
                        else "Assimetria positiva" if dist_sugerida == "Lognormal" 
                        else "Melhor para dados percentuais"}
        """)
    
    # Visualização Q-Q
    st.subheader("Análise de Aderência Distribucional")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    plot_qq(dados_historicos, "Normal", axs[0])
    plot_qq(dados_historicos, "Lognormal", axs[1])
    plot_qq(dados_historicos, "Beta", axs[2])
    st.pyplot(fig)

    # Controles de Simulação
    st.subheader("Configuração da Simulação")
    dist_opcoes = ["Normal", "Beta", "Lognormal", "Bootstrap"]
    dist_selecionada = st.selectbox(
        "Distribuição para Simulação:",
        dist_opcoes,
        index=dist_opcoes.index(dist_sugerida))
    
    num_simulacoes = st.number_input("Número de Simulações", 1000, 100000, 10000, step=1000)
    valor_alvo = st.number_input("Valor Alvo de Multa (R$)", 0.0, 1e9, 50000.0)

    # Execução da Simulação
    if st.button("Executar Simulação Monte Carlo"):
        dados_parques = []
        for parque in df["Windfarm"].unique():
            dados_parque = df[df["Windfarm"] == parque]
            media = dados_parque["Disponibilidade"].mean()
            std = dados_parque["Disponibilidade"].std()
            wtgs = dados_parque["WTGs"].iloc[0]
            
            dados_parques.append({
                "Windfarm": parque,
                "WTGs": wtgs,
                "Média": media,
                "Desvio Padrão": std if not np.isnan(std) else 0.1,
                "Dados Históricos": dados_parque["Disponibilidade"].values
            })
        
        progress_bar = st.progress(0)
        multas_totais = []
        
        for i in range(num_simulacoes):
            multa_total = 0
            for parque in dados_parques:
                disp_simulada = simular_disponibilidade(
                    parque["Média"],
                    parque["Desvio Padrão"],
                    dist_selecionada,
                    parque["Dados Históricos"]
                )
                
                penalidade = max((contrato / disp_simulada) - 1, 0)
                multa = penalidade * fator_multa * parque["WTGs"]
                multa_total += multa
            
            multas_totais.append(multa_total)
            progress_bar.progress((i + 1) / num_simulacoes)
        
        # Resultados
        probabilidade = (np.sum(np.array(multas_totais) > valor_alvo)) / num_simulacoes * 100
        
        st.subheader("Resultados da Simulação")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(f"""
            **Métricas Principais:**
            - Probabilidade de Exceder: **{probabilidade:.2f}%**
            - Média das Multas: R$ {np.mean(multas_totais):,.2f}
            - Mediana: R$ {np.median(multas_totais):,.2f}
            - Valor Máximo: R$ {np.max(multas_totais):,.2f}
            """)
        
        with col_res2:
            fig_dist = plt.figure(figsize=(10, 6))
            plt.hist(multas_totais, bins=50, density=True, alpha=0.6)
            plt.axvline(valor_alvo, color='r', linestyle='--', label='Valor Alvo')
            plt.title("Distribuição das Multas Simuladas")
            plt.xlabel("Valor da Multa (R$)")
            plt.ylabel("Densidade de Probabilidade")
            plt.legend()
            st.pyplot(fig_dist)

if __name__ == "__main__":
    main()
