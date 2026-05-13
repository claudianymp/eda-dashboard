import pandas as pd
import streamlit as st
import numpy as np

@st.cache_data
def load_csv_to_dataframe(data_path):
    df = pd.read_csv(data_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

def load_info_dataset(df):
    if df is not None:
        
        df_info = pd.DataFrame({
            "Coluna": df.columns,
            "Tipo de Dado": [str(t) for t in df.dtypes.values],
            "Valores Não Nulos": df.count().values,
            "Valores Nulos": df.isnull().sum().values
        })
    
    return df_info

def load_statistics_summary(df):
    if df is not None:
        st.dataframe(df.describe().T, width='stretch')
        st.markdown("""
        * **Count:** Total de amostras não nulas.
        * **Mean (Média):** Valor médio aritmético (útil para ver o 'tamanho padrão' do tumor).
        * **Std (Desvio Padrão):** O quanto os dados variam. Valores altos indicam grande diversidade clínica.
        * **Min/Max:** Os limites extremos observados na amostra.
        * **25%, 50%, 75% (Quartis):** Ajudam a identificar a distribuição. O **50% (Mediana)** mostra o valor central, sendo menos sensível a outliers que a média.
        """)  