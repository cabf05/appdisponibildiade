import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import io

st.set_page_config(page_title="Análise de Disponibilidade de Windfarms", layout="wide")

st.title("Análise de Disponibilidade e Cálculo de Multas/Bônus para Windfarms")

# --- Barra Lateral para Inputs ---
st.sidebar.header("Parâmetros de Entrada")
fator_multa = st.sidebar.number_input("Fator Multa", value=1.0, format="%.4f")
disp_contratual = st.sidebar.number_input("Disponibilidade Contratual", value=0.95, format="%.4f")
fator_bonus = st.sidebar.number_input("Fator Bônus", value=1.0, format="%.4f")
pmd = st.sidebar.number_input("PMD", value=1.0, format="%.4f")
mwh_por_wtg = st.sidebar.number_input("MWh por WTG", value=1.0, format="%.4f")

uploaded_file = st.sidebar.file_uploader("Faça upload da planilha Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Ler a planilha
        df = pd.read_excel(uploaded_file)
        st.subheader("Dados Carregados")
        st.dataframe(df)
        
        # Supondo que a primeira coluna é 'Windfarm' e a segunda é 'WTGs'
        # As demais colunas são de disponibilidade no formato "MM/YYYY" ou similar.
        data_columns = df.columns[2:]
        
        # Converter a planilha para formato longo
        records = []
        for idx, row in df.iterrows():
            windfarm = row[df.columns[0]]
            wtgs = row[df.columns[1]]
            for col in data_columns:
                # Tentar extrair mês e ano a partir do nome da coluna
                # Exemplo esperado: "01/2018", "02/2018", ou "Jan-2018"
                try:
                    # Primeiro, tenta com o separador '/'
                    mes, ano = str(col).split("/")
                    mes = mes.strip()
                    ano = ano.strip()
                except Exception:
                    try:
                        # Tenta com o separador '-'
                        mes, ano = str(col).split("-")
                        mes = mes.strip()
                        ano = ano.strip()
                    except Exception:
                        # Se não conseguir, ignora ou define como None
                        mes, ano = None, None
                dispon = row[col]
                records.append({
                    "Windfarm": windfarm,
                    "WTGs": wtgs,
                    "Mes": mes,
                    "Ano": ano,
                    "Disponibilidade": dispon
                })
        
        df_long = pd.DataFrame(records)
        # Converter o campo Ano para string (caso não esteja) e Disponibilidade para float
        df_long["Ano"] = df_long["Ano"].astype(str)
        df_long["Disponibilidade"] = pd.to_numeric(df_long["Disponibilidade"], errors="coerce")
        st.subheader("Dados no Formato Longo")
        st.dataframe(df_long.head(20))
        
        # --- Cálculos Estatísticos Anuais ---
        st.header("Estatísticas Anuais por Windfarm")
        stats_df = df_long.groupby(["Windfarm", "Ano"]).agg(
            Media_Anual=("Disponibilidade", "mean"),
            Mediana=("Disponibilidade", "median"),
            Desvio_Padrao=("Disponibilidade", "std"),
            Qtd_Medicoes=("Disponibilidade", "count")
        ).reset_index()
        st.dataframe(stats_df)
        
        # --- Função para Ajuste de Distribuição ---
        def ajustar_distribuicao(data):
            # Lista de distribuições candidatas
            distros = {
                "Normal": stats.norm,
                "Lognormal": stats.lognorm,
                "Gama": stats.gamma
            }
            resultados = {}
            for nome, dist in distros.items():
                try:
                    if nome == "Lognormal":
                        # Para lognormal, os dados precisam ser positivos
                        data_pos = data[data > 0]
                        if len(data_pos) == 0:
                            continue
                        params = dist.fit(data_pos)
                        # Teste KS
                        ks_stat, p_value = stats.kstest(data_pos, nome.lower(), args=params)
                    else:
                        params = dist.fit(data)
                        ks_stat, p_value = stats.kstest(data, nome.lower(), args=params)
                    resultados[nome] = {"p_value": p_value, "params": params}
                except Exception as e:
                    resultados[nome] = {"p_value": 0, "params": None}
            if resultados:
                melhor = max(resultados.items(), key=lambda x: x[1]["p_value"])
                return melhor[0], resultados[melhor[0]]
            else:
                return None, None

        # --- Cálculo de Probabilidades e Multas/Bônus ---
        st.header("Análises de Probabilidade e Multas/Bônus")
        resultados = []
        windfarms = df_long["Windfarm"].unique()
        for wf in windfarms:
            df_wf = df_long[df_long["Windfarm"] == wf]
            # Pegando a série mensal de disponibilidades
            data_mensal = df_wf["Disponibilidade"].dropna()
            dist_nome, dist_info = ajustar_distribuicao(data_mensal)
            # Se o melhor ajuste tiver p-value baixo, usamos método empírico
            metodo = ""
            p_valor_fit = None
            if dist_info is not None:
                p_valor_fit = dist_info["p_value"]
            if dist_info is None or p_valor_fit < 0.05:
                metodo = "Empírico"
                # Probabilidade empírica de um mês ficar abaixo da Disponibilidade Contratual
                prob_mes = np.mean(data_mensal < disp_contratual)
                # Para o anual, como a média é a soma de 12 meses, calculamos empiricamente
                # Agrupar por ano para cada windfarm já foi feito: usaremos os valores da média anual
                df_medias = df_wf.groupby("Ano")["Disponibilidade"].mean()
                prob_ano = np.mean(df_medias < disp_contratual)
                grau_confianca = "Baixo (método empírico com poucos dados)"
            else:
                metodo = dist_nome
                params = dist_info["params"]
                # Usar a função CDF da distribuição ajustada para calcular a probabilidade de um mês abaixo do valor contratual
                if dist_nome == "Normal":
                    prob_mes = stats.norm.cdf(disp_contratual, *params)
                elif dist_nome == "Lognormal":
                    prob_mes = stats.lognorm.cdf(disp_contratual, *params)
                elif dist_nome == "Gama":
                    prob_mes = stats.gamma.cdf(disp_contratual, *params)
                else:
                    prob_mes = None
                # Para o anual, utilizamos simulação de Monte Carlo
                sim_n = 10000
                sim_meses = None
                if dist_nome == "Normal":
                    sim_meses = stats.norm.rvs(*params, size=(sim_n, 12))
                elif dist_nome == "Lognormal":
                    sim_meses = stats.lognorm.rvs(*params, size=(sim_n, 12))
                elif dist_nome == "Gama":
                    sim_meses = stats.gamma.rvs(*params, size=(sim_n, 12))
                if sim_meses is not None:
                    sim_anuais = sim_meses.mean(axis=1)
                    prob_ano = np.mean(sim_anuais < disp_contratual)
                grau_confianca = f"p-valor do ajuste: {p_valor_fit:.3f}"
            
            # Para cada ano deste windfarm, calcular multa ou bônus
            df_anual = df_wf.groupby("Ano").agg(
                Media_Anual=("Disponibilidade", "mean"),
                WTGs=("WTGs", "first")
            ).reset_index()
            multas_bonos = []
            for _, row in df_anual.iterrows():
                media_anual = row["Media_Anual"]
                wtgs = row["WTGs"]
                if media_anual < disp_contratual:
                    multa = fator_multa * pmd * mwh_por_wtg * wtgs * ((disp_contratual / media_anual) - 1)
                    bonus = 0
                else:
                    bonus = fator_bonus * pmd * mwh_por_wtg * wtgs * ((media_anual / disp_contratual) - 1)
                    multa = 0
                multas_bonos.append({
                    "Windfarm": wf,
                    "Ano": row["Ano"],
                    "Media_Anual": media_anual,
                    "WTGs": wtgs,
                    "Multa": multa,
                    "Bônus": bonus,
                    "Prob_Ano < Disp_Contratual": prob_ano,
                    "Metodo_Ajuste": metodo,
                    "Grau_Confianca": grau_confianca,
                    "Prob_Mes < Disp_Contratual": prob_mes
                })
            resultados.extend(multas_bonos)
        
        resultados_df = pd.DataFrame(resultados)
        st.subheader("Resultados de Probabilidade e Cálculo de Multas/Bônus")
        st.dataframe(resultados_df)
        
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")

else:
    st.info("Por favor, faça o upload de um arquivo Excel para iniciar a análise.")
