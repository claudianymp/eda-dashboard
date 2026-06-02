import streamlit as st
from src.kdd.selecao_integracao import load_csv_to_dataframe, mostrar_informacao_dataset, iniciar_etapa_selecao_integracao, mostrar_estrutura_dataset
from src.kdd.pre_processamento import iniciar_etapa_pre_processamento
from src.kdd.transformacacao import iniciar_etapa_transformacao_pca, visualizar_pca_3d
from src.kdd.mineracao import mineracao_dbscan, mineracao_rede_neural

st.set_page_config(
    page_title='EDA - Dashboard',
    page_icon=':bar_chart:',
    layout='wide'
)

st.title("Dashboard - Análise Exploratória de Dados (EDA)")
st.markdown("---")

tab_kdd_flow, tab_analysis,tab_train_model  = st.tabs([
    "Fluxo KDD (Processo de Descoberta)",
    "Análise Exploratória de Dados",
    "Treinar Rede Neural"
])

st.session_state.df_test = None
st.session_state.df_train = None
st.session_state.X_train_scaled = None
st.session_state.X_test_scaled = None
st.session_state.df_train_pca = None 
st.session_state.df_test_pca = None 
st.session_state.pipe = None

with tab_kdd_flow:
    with st.expander("Seleção e Integração dos dados"):
        df_test = load_csv_to_dataframe('./data/test.csv')
        df_train = load_csv_to_dataframe('./data/train.csv')
        
        if(df_test is not None):
            st.session_state.df_test = iniciar_etapa_selecao_integracao(df_test)
        
        if(df_train is not None):
            st.session_state.df_train = iniciar_etapa_selecao_integracao(df_train)
            
        with st.expander("Visualização do dataset selecionado: Colunas mantidas"):
            mostrar_estrutura_dataset(st.session_state.df_train, 10)
        
        with st.expander("Informações do dataset: colunas, tipos de dados, dados não nulos e nulos"):
            st.table(mostrar_informacao_dataset(st.session_state.df_train))
    
    with st.expander("Pré-processamento dos dados"):
        df_train, df_test, X_train_scaled, X_test_scaled = iniciar_etapa_pre_processamento(st.session_state.df_train, st.session_state.df_test)
        st.session_state.df_train = df_train
        st.session_state.df_test = df_test
        st.session_state.X_train_scaled = X_train_scaled
        st.session_state.X_test_scaled = X_test_scaled
            
    with st.expander("Transformação: Redução de Dimensionalidade (PCA)"):
        df_train_pca, df_test_pca, pipe = iniciar_etapa_transformacao_pca(
            st.session_state.df_train, 
            st.session_state.df_test, 
            n_componentes=3
        )
        st.session_state.df_train_pca = df_train_pca 
        st.session_state.df_test_pca = df_test_pca 
        st.session_state.pipe = pipe
        
        visualizar_pca_3d(st.session_state.df_train_pca, st.session_state.df_train, 'Diagnóstico')

    with st.expander("Mineração: Agrupamento Não Supervisionado (DBSCAN)"):
        if "df_train_pca" in st.session_state:
            df_com_clusters = mineracao_dbscan(st.session_state.df_train_pca)
            st.session_state.df_train_pca = df_com_clusters
            
            st.write("Amostra dos dados agrupados:")
            st.dataframe(df_com_clusters.head())
            
            visualizar_pca_3d(st.session_state.df_train_pca, st.session_state.df_train, 'Cluster')
        
    with st.expander("Mineração: Agrupamento Supervisionado (MLP)"):
        st.write("Dados agrupados lalala")
        # model = mineracao_rede_neural(st.session_state.df_train_pca, st.session_state.df_test_pca)
        # st.dataframe(model)
                    
with tab_train_model:
    st.subheader('Treinar modelo')
    
with tab_analysis:
    st.subheader('Análise de Dataset Tabular')
    

