import pandas as pd
import plotly.express as px
import streamlit as st

def exibir_preparacao_mineracao(
    df_original,
    df_train,
    df_test,
    df_train_pca,
    df_test_pca,
    pipe_pca
):
    st.header("Preparação para Mineração")

    with st.expander(
        "Seleção de Atributos",
        expanded=True
    ):
        atributos = [
            c for c in df_original.columns
            if c not in [
                'cancer_presence',
                'cancer_subtype'
            ]
        ]

        st.write(
            f"Total de atributos selecionados: {len(atributos)}"
        )

        st.dataframe(
            pd.DataFrame({
                "Atributo": atributos
            }),
            use_container_width=True
        )

    with st.expander(
        "Qualidade após Pré-processamento"
    ):
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Nulos Treino",
                int(df_train.isnull().sum().sum())
            )

        with col2:
            st.metric(
                "Nulos Teste",
                int(df_test.isnull().sum().sum())
            )

    with st.expander(
        "Distribuição da Classe Alvo"
    ):
        distribuicao = (
            df_original['cancer_presence']
            .value_counts()
            .reset_index()
        )

        distribuicao.columns = [
            "Classe",
            "Quantidade"
        ]

        distribuicao["Classe"] = (
            distribuicao["Classe"]
            .map({
                0: "Não Detectado",
                1: "Detectado"
            })
        )

        fig = px.bar(
            distribuicao,
            x="Classe",
            y="Quantidade",
            color="Classe",
            text="Quantidade",
            title="Distribuição da Classe Alvo"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with st.expander(
        "Divisão Treino/Teste"
    ):
        col1, col2 = st.columns(2)
        col1.metric(
            "Registros Treino",
            len(df_train)
        )
        col2.metric(
            "Registros Teste",
            len(df_test)
        )

    with st.expander(
        "Redução de Dimensionalidade (PCA)"
    ):
        pca = pipe_pca.named_steps['pca']
        variancia = (
            pca.explained_variance_ratio_
            * 100
        )

        df_variancia = pd.DataFrame({
            "Componente":
                [f"PC{i+1}" for i in range(len(variancia))],
            "Variância (%)":
                variancia
        })

        fig = px.bar(
            df_variancia,
            x="Componente",
            y="Variância (%)",
            text="Variância (%)",
            title="Variância Explicada pelos Componentes"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            df_variancia,
            use_container_width=True
        )
        
    with st.expander(
        "Dataset Preparado para Mineração"
    ):
        st.write(
            f"Shape Treino PCA: {df_train_pca.shape}"
        )
        st.write(
            f"Shape Teste PCA: {df_test_pca.shape}"
        )
        st.dataframe(
            df_train_pca.head(),
            use_container_width=True
        )

    with st.expander(
        "7️Resumo da Preparação",
        expanded=True
    ):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Features Originais",
            len(df_train.columns)
        )

        col2.metric(
            "Componentes PCA",
            df_train_pca.shape[1]
        )

        col3.metric(
            "Treino",
            len(df_train)
        )

        col4.metric(
            "Teste",
            len(df_test)
        )

        st.success("Dados preparados com sucesso para as etapas de mineração (DBSCAN e MLP).")