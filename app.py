import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

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

def infer_date_format(col_name: str):
    """
    Inferir o formato da data com base na posição do '01'.
    Se o primeiro token for '01', assume-se formato brasileiro (dia/mês/ano).
    Se o segundo token for '01', assume-se formato americano (mês/dia/ano).
    Retorna uma string "BR" ou "US" ou None se não for possível inferir.
    """
    if "/" in col_name:
        tokens = col_name.split("/")
    elif "-" in col_name:
        tokens = col_name.split("-")
    else:
        return None
    if len(tokens) != 3:
        return None
    if tokens[0].strip() == "01":
        return "BR"
    elif tokens[1].strip() == "01":
        return "US"
    else:
        return None

def parse_date_from_col(col_name: str):
    """
    Extrai dia, mês e ano do nome da coluna, considerando os formatos esperados.
    Retorna dia, mês e ano (como strings).
    """
    fmt = infer_date_format(col_name)
    # Normaliza separador para '/'
    col_name = col_name.replace("-", "/")
    tokens = col_name.split("/")
    if len(tokens) != 3:
        return None, None, None
    if fmt == "BR":
        # Formato brasileiro: dia/mês/ano
        day, mes, ano = tokens[0].strip(), tokens[1].strip(), tokens[2].strip()
    elif fmt == "US":
        # Formato americano: mês/dia/ano
        day, mes, ano = tokens[1].strip(), tokens[0].strip(), tokens[2].strip()
    else:
        # Se não conseguir inferir, tenta converter usando o pandas com dayfirst=True
        try:
            dt = pd.to_datetime(col_name, dayfirst=True)
            day, mes, ano = str(dt.day), str(dt.month), str(dt.year)
        except Exception:
            day, mes, ano = None, None, None
    return day, mes, ano

if uploaded_file is not None:
    try:
        # Ler a planilha
        df = pd.read_excel(uploaded_file)
        st.subheader("Dados Carregados")
        st.dataframe(df)
        
        # Supomos que a primeira coluna é 'Windfarm' e a segunda é 'WTGs'
        # As demais colunas possuem datas no formato "01/mês/ano" ou "mês/01/ano".
        data_columns = df.columns[2:]
        
        # Converter a planilha para formato longo
        records = []
        for idx, row in df.iterrows():
            windfarm = row[df.columns[0]]
            wtgs = row[df.columns[1]]
            for col in data_columns:
                day, mes, ano = parse_date_from_col(str(col))
                dispon = row[col]
                records.append({
                    "Windfarm": windfarm,
                    "WTGs": wtgs,
                    "Dia": day,
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
            metodo = ""
            p_valor_fit = None
            if dist_info is not None:
                p_valor_fit = dist_info["p_value"]
            if dist_info is None or p_valor_fit < 0.05:
                metodo = "Empírico"
                prob_mes = np.mean(data_mensal < disp_contratual)
                df_medias = df_wf.groupby("Ano")["Disponibilidade"].mean()
                prob_ano = np.mean(df_medias < disp_contratual)
                grau_confianca = "Baixo (método empírico com poucos dados)"
            else:
                metodo = dist_nome
                params = dist_info["params"]
                if dist_nome == "Normal":
                    prob_mes = stats.norm.cdf(disp_contratual, *params)
                elif dist_nome == "Lognormal":
                    prob_mes = stats.lognorm.cdf(disp_contratual, *params)
                elif dist_nome == "Gama":
                    prob_mes = stats.gamma.cdf(disp_contratual, *params)
                else:
                    prob_mes = None
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
