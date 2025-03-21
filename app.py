import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Configuração da página
st.set_page_config(page_title="Análise de Disponibilidade de Parques Eólicos", layout="wide")

# Função para processar a planilha
def processar_dados(arquivo):
    df = pd.read_excel(arquivo, sheet_name="Sheet1")
    
    # Converter colunas de datas para formato longo
    id_vars = ["Windfarm", "WTGs"]
    value_vars = df.columns[2:]
    df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="Data", value_name="Disponibilidade")
    
    # Extrair ano e mês
    df_melted["Data"] = pd.to_datetime(df_melted["Data"])
    df_melted["Ano"] = df_melted["Data"].dt.year
    df_melted["Mês"] = df_melted["Data"].dt.month
    
    return df_melted

# Carregar arquivo
arquivo = st.file_uploader("Carregue a planilha de disponibilidade (formato idêntico ao exemplo)", type="xlsx")
if not arquivo:
    st.stop()

df = processar_dados(arquivo)

# Tabela 1: Número de WTGs por Parque e Total
st.header("1. Número de WTGs por Parque Eólico")
tabela_wtgs = df.groupby("Windfarm")["WTGs"].first().reset_index()
total_wtgs = tabela_wtgs["WTGs"].sum()
st.markdown(f"**Total Geral de WTGs:** {total_wtgs}")
st.dataframe(tabela_wtgs)

# Tabela 2: Disponibilidade por Ano e Mês por Parque
st.header("2. Disponibilidade por Ano e Mês")
tabela_disponibilidade = df.pivot_table(
    index=["Windfarm", "Ano"],
    columns="Mês",
    values="Disponibilidade",
    aggfunc="mean"
).reset_index()
st.dataframe(tabela_disponibilidade)

# Tabela 3: Disponibilidade Média por Ano por Parque
st.header("3. Disponibilidade Média por Ano")
tabela_media_anual = df.groupby(["Windfarm", "Ano"])["Disponibilidade"].mean().reset_index()
st.dataframe(tabela_media_anual)

# Tabela 4: Desvio Padrão por Ano por Parque
st.header("4. Desvio Padrão por Ano")
tabela_std_anual = df.groupby(["Windfarm", "Ano"])["Disponibilidade"].std().reset_index()
st.dataframe(tabela_std_anual)

# Tabela 5: Cálculo (Contrato / Média Anual) - 1
st.header("5. Cálculo de Penalidade Base")
contrato = st.number_input("Disponibilidade de Contrato (%)", min_value=0.0, max_value=100.0, value=95.0)
merged = pd.merge(tabela_media_anual, tabela_wtgs, on="Windfarm")
merged["Tabela5"] = (contrato / merged["Disponibilidade"] - 1).round(4)
merged["Tabela5"] = merged["Tabela5"].apply(lambda x: max(x, 0))  # Substituir negativos por 0
st.dataframe(merged[["Windfarm", "Ano", "Tabela5"]])

# Tabela 6: Cálculo da Multa
st.header("6. Cálculo da Multa por Parque")
fator_multa = st.number_input("Fator de Multa (R$ por WTG)", min_value=0.0, value=1000.0)
merged["Tabela6"] = merged["Tabela5"] * fator_multa * merged["WTGs"]
st.dataframe(merged[["Windfarm", "Ano", "Tabela6"]])

# Soma Total de Multas por Ano
st.header("7. Soma Total de Multas por Ano")
soma_multa_ano = merged.groupby("Ano")["Tabela6"].sum().reset_index()
st.dataframe(soma_multa_ano)

# Probabilidade Empírica de Multa > Valor Alvo
st.header("8. Probabilidade de Multa Total > Valor Alvo")
valor_alvo = st.number_input("Valor Alvo (R$)", min_value=0.0, value=50000.0)
historico = soma_multa_ano["Tabela6"].values
excedeu = sum(historico > valor_alvo)
probabilidade = excedeu / len(historico) * 100 if len(historico) > 0 else 0
st.markdown(f"**Probabilidade Empírica:** {probabilidade:.2f}%")
