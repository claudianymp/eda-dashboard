import streamlit as st
import pandas as pd
from src.load_data import load_csv_to_dataframe, load_info_dataset

st.set_page_config(
    page_title='EDA - Dashboard',
    page_icon=':bar_chart:',
    layout='wide'
)

def show_data_information(df, limit=5):
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

st.title("Dashboard - Análise Exploratória de Dados (EDA)")
st.markdown("---")

tab_loading_data, tab_analysis,tab_train_model  = st.tabs([
    "Explorar Dataset",
    "Análise Exploratória de Dados",
    "Treinar Rede Neural"
])

st.session_state.df_test = None
st.session_state.df_train = None
st.session_state.df = None
with tab_loading_data:
    with st.expander("Carregar Dados", expanded=True):
        st.session_state.df_test = load_csv_to_dataframe('./data/test.csv')
        st.session_state.df_train = load_csv_to_dataframe('./data/train.csv')
        st.session_state.df = pd.concat([st.session_state.df_train, st.session_state.df_test], ignore_index=True)
        if(st.session_state.df is not None):
            st.success(f'Dados carregados! Total de linhas: {len(st.session_state.df)}')
            
    with st.expander("Visualização de amostras do dataset"):
        show_data_information(st.session_state.df, 10)
        
    with st.expander("Informações do dataset: colunas, tipos de dados, dados não nulos e nulos"):
        st.table(load_info_dataset(st.session_state.df))
                    
with tab_analysis:
    st.subheader('Análise de Dataset Tabular')
    
with tab_train_model:
    st.subheader('Treinar modelo')
