import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path

@st.cache_data
def load_csv_to_dataframe(data_path):
    df = pd.read_csv(data_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    df.attrs["dataset_name"] = Path(data_path).name.upper()
    
    return df

def iniciar_etapa_selecao_integracao(df):
    features_radiomicas = ['nodule_size_mm', 'HU_mean', 'HU_std', 'PET_SUVmax', 'cancer_stage']
    features_clinicas = ['patient_age', 'smoking_history', 'tumor_location', 
                         'patient_gender', 'family_history']
    target = ['cancer_presence', 'cancer_subtype']
    
    df_selecionado = df[features_radiomicas + features_clinicas + target]
        
    return df_selecionado


def mostrar_informacao_dataset(df):
    if df is not None:
        
        df_info = pd.DataFrame({
            "Coluna": df.columns,
            "Tipo de Dado": [str(t) for t in df.dtypes.values],
            "Valores Não Nulos": df.count().values,
            "Valores Nulos": df.isnull().sum().values
        })
    
    return df_info

def mostrar_estrutura_dataset(df, limit=5):
    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Linhas", df.shape[0])
        col2.metric("Total de Colunas", df.shape[1])
        col3.metric("Dados Faltantes", df.isna().sum().sum())

        tab1, tab2 = st.tabs(["Início (Head)", "Fim (Tail)"])
        with tab1:
            st.subheader("Registros iniciais do dataset")
            st.dataframe(df.head(limit), width='stretch')
        
        with tab2:
            st.subheader("Registros finais do dataset")
            st.dataframe(df.tail(limit), width='stretch')
            
def mostrar_resumo_estatistico(df):
    if df is not None:
        st.dataframe(df.describe().T, width='stretch')
        st.markdown("""
        * **Count:** Total de amostras não nulas.
        * **Mean (Média):** Valor médio aritmético (útil para ver o 'tamanho padrão' do tumor).
        * **Std (Desvio Padrão):** O quanto os dados variam. Valores altos indicam grande diversidade clínica.
        * **Min/Max:** Os limites extremos observados na amostra.
        * **25%, 50%, 75% (Quartis):** Ajudam a identificar a distribuição. O **50% (Mediana)** mostra o valor central, sendo menos sensível a outliers que a média.
        """)  