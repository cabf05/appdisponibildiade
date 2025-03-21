import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Configuração da página
st.set_page_config(page_title="Análise de Parques Eólicos", layout="wide")

# Função para processar dados
def processar_dados(arquivo):
    df = pd.read_excel(arquivo, sheet_name="Sheet1")
    
    # Converter para formato longo
    id_vars = ["Windfarm", "WTGs"]
    value_vars = df.columns[2:]
    df_melted = df.melt(
        id_vars=id_vars, 
        value_vars=value_vars, 
        var_name="Data", 
        value_name="Disponibilidade"
    )
    
    # Extrair datas
    df_melted["Data"] = pd.to_datetime(df_melted["Data"])
    df_melted["Ano"] = df_melted["Data"].dt.year
    df_melted["Mês"] = df_melted["Data"].dt.month
    
    return df_melted

# Interface principal
def main():
    st.title("Sistema de Análise de Disponibilidade de Parques Eólicos")
    
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
    ).reset_index()
    st.dataframe(tabela_mensal.style.format("{:.2f}"), use_container_width=True)

    # Tabela 3: Média Anual
    st.header("3. Disponibilidade Média Anual")
    tabela_media_anual = df.groupby(["Windfarm", "Ano"])["Disponibilidade"].mean().reset_index()
    st.dataframe(tabela_media_anual, use_container_width=True)

    # Tabela 4: Desvio Padrão Anual
    st.header("4. Desvio Padrão Anual")
    tabela_std_anual = df.groupby(["Windfarm", "Ano"])["Disponibilidade"].std().reset_index()
    st.dataframe(tabela_std_anual, use_container_width=True)

    # Tabela 5: Cálculo de Penalidade Base
    st.header("5. Cálculo de Penalidade")
    contrato = st.number_input("Disponibilidade de Contrato (%)", 
                             min_value=0.0, 
                             max_value=100.0, 
                             value=95.0,
                             key="contrato")
    
    # Merge dos dados
    merged = pd.merge(tabela_media_anual, tabela_wtgs, on="Windfarm")
    merged["Tabela5"] = (contrato / merged["Disponibilidade"] - 1).round(4)
    merged["Tabela5"] = merged["Tabela5"].apply(lambda x: max(x, 0))
    st.dataframe(merged[["Windfarm", "Ano", "Tabela5"]], use_container_width=True)

    # Tabela 6: Cálculo da Multa
    st.header("6. Cálculo da Multa")
    fator_multa = st.number_input("Fator de Multa (R$ por WTG)", 
                                 min_value=0.0, 
                                 value=1000.0,
                                 key="fator_multa")
    
    merged["Tabela6"] = merged["Tabela5"] * fator_multa * merged["WTGs"]
    st.dataframe(merged[["Windfarm", "Ano", "Tabela6"]], use_container_width=True)

    # Soma Total de Multas
    st.header("7. Soma Total de Multas por Ano")
    soma_multa_ano = merged.groupby("Ano")["Tabela6"].sum().reset_index()
    st.dataframe(soma_multa_ano, use_container_width=True)

    # Simulação Monte Carlo
    st.header("8. Simulação de Probabilidade de Multa")
    col1, col2 = st.columns(2)
    with col1:
        valor_alvo = st.number_input("Valor Alvo (R$)", 
                                    min_value=0.0, 
                                    value=50000.0,
                                    key="valor_alvo")
    with col2:
        num_simulacoes = st.number_input("Número de Simulações", 
                                        min_value=1000, 
                                        value=10000,
                                        step=1000,
                                        key="simulacoes")

    # Preparar dados para simulação
    dados_parques = []
    for parque in tabela_wtgs["Windfarm"].unique():
        wtgs = tabela_wtgs[tabela_wtgs["Windfarm"] == parque]["WTGs"].values[0]
        media = tabela_media_anual[tabela_media_anual["Windfarm"] == parque]["Disponibilidade"].mean()
        std = tabela_std_anual[tabela_std_anual["Windfarm"] == parque]["Disponibilidade"].mean()
        
        dados_parques.append({
            "Windfarm": parque,
            "WTGs": wtgs,
            "Média": media,
            "Desvio Padrão": std if not np.isnan(std) else 0.1  # Evitar NaN
        })

    # Executar simulação
    if st.button("Executar Simulação"):
        multas_totais = []
        progress_bar = st.progress(0)
        
        for i in range(num_simulacoes):
            multa_total = 0
            for parque in dados_parques:
                # Simular disponibilidade com distribuição normal truncada
                disp_simulada = np.random.normal(parque["Média"], parque["Desvio Padrão"])
                disp_simulada = np.clip(disp_simulada, 0, 100)
                
                # Calcular multa
                penalidade = max((contrato / disp_simulada) - 1, 0)
                multa = penalidade * fator_multa * parque["WTGs"]
                multa_total += multa
                
            multas_totais.append(multa_total)
            progress_bar.progress((i + 1) / num_simulacoes)

        # Calcular probabilidade
        probabilidade = (np.sum(np.array(multas_totais) > valor_alvo) / num_simulacoes * 100
        
        # Exibir resultados
        st.markdown(f"""
        **Resultados da Simulação:**
        - Probabilidade de exceder R$ {valor_alvo:,.2f}: **{probabilidade:.2f}%**
        - Média simulada das multas: R$ {np.mean(multas_totais):,.2f}
        - Maior multa simulada: R$ {np.max(multas_totais):,.2f}
        """)

        # Plotar histograma
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(multas_totais, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(valor_alvo, color='red', linestyle='--', label='Valor Alvo')
        ax.set_xlabel("Valor Total da Multa (R$)")
        ax.set_ylabel("Frequência")
        ax.set_title("Distribuição das Multas Simuladas")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
