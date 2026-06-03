import streamlit as st
import pandas as pd
from src.kdd.selecao_integracao import load_csv_to_dataframe, mostrar_informacao_dataset, iniciar_etapa_selecao_integracao, mostrar_estrutura_dataset
from src.kdd.pre_processamento import iniciar_etapa_pre_processamento
from src.kdd.transformacacao import iniciar_etapa_transformacao_pca, visualizar_pca_3d
from src.kdd.mineracao import mineracao_dbscan, mineracao_rede_neural
from src.kdd.dashboard_avaliacao import dashboard_avaliacao_modelo
from src.eda.visao_geral import montar_visao_geral, plot_distribuicao_geral_diagnostico
from src.eda.analise_perfil_pacientes import graficos_cancer_por_genero, graficos_cancer_por_idade, graficos_cancer_por_estagio
st.set_page_config(
    page_title='EDA - Dashboard',
    page_icon=':bar_chart:',
    layout='wide'
)

st.title("Dashboard - Análise Exploratória de Dados (EDA)")
st.markdown("---")

tab_eda, tab_kdd = st.tabs(["EDA - Análise Exploratória de Dados", "Fluxo KDD"])

st.session_state.df_test = None
st.session_state.df_train = None

with tab_eda:
    st.subheader('Análise de Dataset Tabular')
    
    tab_visao_geral, tab_perfil_pacientes, tab_caract_clinicas, tab_dist_estatistica, tab_qual_dados, tab_prep_mineracao  = st.tabs([
        "Visão Geral do Dataset",
        "1. Perfil Epidemiológico dos Pacientes",
        "2. Características Clínicas",
        "3. Distribuições Estatísticas",
        "4. Qualidade dos Dados",
        "5. Preparação para Mineração",
    ])
    
    df_test = load_csv_to_dataframe('./data/test.csv')
    df_train = load_csv_to_dataframe('./data/train.csv')

    if(df_test is not None):
        st.session_state.df_test = iniciar_etapa_selecao_integracao(df_test)

    if(df_train is not None):
        st.session_state.df_train = iniciar_etapa_selecao_integracao(df_train)
    
    with tab_visao_geral:
        montar_visao_geral(st.session_state.df_train)
        plot_distribuicao_geral_diagnostico(st.session_state.df_train)
        
    with tab_perfil_pacientes:
        with st.expander("Diagnóstico por Gênero"):
            graficos_cancer_por_genero(st.session_state.df_train)
        
        with st.expander("Diagnóstico por Idade"):
            graficos_cancer_por_idade(st.session_state.df_train)
            
        with st.expander("Diagnóstico por Estágio"):
            graficos_cancer_por_estagio(st.session_state.df_train)
    
    with tab_caract_clinicas:
        st.write("lalal")
    
    with tab_dist_estatistica:
        st.write("lalal")
        
    with tab_qual_dados:
        st.write("lalal")
        
    with tab_prep_mineracao:
        st.write("lalal")

with tab_kdd:
    tab_selecao, tab_pre_processamento, tab_transformacao, tab_mineracao, tab_resultado  = st.tabs([
        "1. Seleção e Integração dos dados",
        "2. Pré-processamento dos dados",
        "3. Transformação",
        "4. Mineração",
        "5. Avaliação e Interpretação",
    ])

    st.session_state.X_train_scaled = None
    st.session_state.X_test_scaled = None
    st.session_state.df_train_pca = None 
    st.session_state.df_test_pca = None 
    st.session_state.pipe = None

    with tab_selecao:
        st.success(f"Seleção concluída - dataset {st.session_state.df_train.attrs['dataset_name']} : {st.session_state.df_train.shape[1]} colunas mantidas.")
        st.success(f"Seleção concluída - dataset {st.session_state.df_test.attrs['dataset_name']} : {st.session_state.df_test.shape[1]} colunas mantidas.")

        with st.expander("Visualização do dataset selecionado: Colunas mantidas"):
            mostrar_estrutura_dataset(st.session_state.df_train, 10)
        
        with st.expander("Informações do dataset: colunas, tipos de dados, dados não nulos e nulos"):
            st.table(mostrar_informacao_dataset(st.session_state.df_train))

    with tab_pre_processamento:
        st.subheader("Tratamento de variáveis categóricas, de valores nulos e normalização")
        (
            st.session_state.df_train, 
            st.session_state.df_test, 
            st.session_state.X_train_scaled, 
            st.session_state.X_test_scaled
        ) = iniciar_etapa_pre_processamento(st.session_state.df_train, st.session_state.df_test)
            
    with tab_transformacao:
        st.subheader("Redução de Dimensionalidade (PCA)")
        (
            st.session_state.df_train_pca,
            st.session_state.df_test_pca,
            st.session_state.pipe_pca
        ) = iniciar_etapa_transformacao_pca(st.session_state.df_train, st.session_state.df_test, n_componentes=3 )
        
        visualizar_pca_3d(st.session_state.df_train_pca, st.session_state.df_train, 'Diagnóstico')

    with tab_mineracao:
        with st.expander("Agrupamento Não Supervisionado (DBSCAN)"):
            if "df_train_pca" in st.session_state:
                (st.session_state.df_train_pca) = mineracao_dbscan(st.session_state.df_train_pca)
                
                st.write("Amostra dos dados agrupados:")
                st.dataframe(st.session_state.df_train_pca.head())
                
                visualizar_pca_3d(st.session_state.df_train_pca, st.session_state.df_train, 'Cluster')
        
        with st.expander("Classificação Supervisionada (MLP)"):
            if ('df_train_pca' in st.session_state and 'df_test_pca' in st.session_state):
                model, loss_history, preds, probs = mineracao_rede_neural(
                    X_train=st.session_state.df_train_pca,
                    y_train=st.session_state.df_train['cancer_presence'],
                    X_test=st.session_state.df_test_pca
                )
                st.line_chart(loss_history)
                
                col1,col2 = st.columns(2)
                col1.subheader("Predições")
                col2.subheader("Probabilidades")
                
                col1.dataframe(
                    pd.DataFrame(
                        preds[:10],
                        columns=['Predito']
                    )
                )
                col2.dataframe(
                    pd.DataFrame(
                    probs[:10],
                    columns=['Probabilidade']
                    )
                )
            else:
                st.error("Execute a etapa de PCA antes do treinamento.")

    with tab_resultado:
        dashboard_avaliacao_modelo(
            pipe_pca=st.session_state.pipe_pca,
            X_test_pca=st.session_state.df_test_pca,
            y_true=st.session_state.df_test['cancer_presence'],
            preds=preds,
            probs=probs,
            dbscan_labels=st.session_state.get('dbscan_labels',None)
        )
                         
   
    

