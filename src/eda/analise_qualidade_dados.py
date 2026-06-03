import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

def visao_dados_qualidade_dados(df):
    st.subheader("Os dados possuem problemas que podem impactar a análise e o treinamento dos modelos?")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Registros",
        len(df)
    )
    col2.metric(
        "Atributos",
        len(df.columns)
    )
    col3.metric(
        "Valores Nulos",
        int(df.isnull().sum().sum())
    )
    col4.metric(
        "Duplicados",
        int(df.duplicated().sum())
    )
    
    nulos = pd.DataFrame({
        "Coluna": df.columns,
        "Nulos": df.isnull().sum()
    })

    nulos = nulos[nulos["Nulos"] > 0].sort_values("Nulos",ascending=False)
    if not nulos.empty:
        fig = px.bar(
            nulos,
            x="Nulos",
            y="Coluna",
            orientation="h",
            text="Nulos",
            title="Valores Nulos por Coluna"
        )

        st.plotly_chart(fig,use_container_width=True)
    else:
        st.success("Nenhum valor nulo encontrado.")
        
    duplicados = df.duplicated().sum()
    if duplicados > 0:
        st.warning(f"Foram encontrados {duplicados} registros duplicados.")
    else:
        st.success("Nenhum registro duplicado encontrado.")
              
   
    tipos = pd.DataFrame({
        "Coluna": df.columns,
        "Tipo": df.dtypes.astype(str)
    })
    st.dataframe(tipos, use_container_width=True)        
    
    colunas = [
        'patient_age',
        'nodule_size_mm',
        'PET_SUVmax',
        'HU_mean',
        'HU_std'
    ]
    variavel = st.selectbox(
        "Variável",
        colunas
    )

    fig = px.box(
        df,
        y=variavel,
        title=f"Outliers - {variavel}"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    total_celulas = (
        len(df)
        * len(df.columns)
    )

    percentual_nulos = (
        df.isnull().sum().sum()
        / total_celulas
    ) * 100

    percentual_duplicados = (
        df.duplicated().sum()
        / len(df)
    ) * 100

    qualidade = (
        100
        - percentual_nulos
        - percentual_duplicados
    )

    qualidade = max(
        0,
        round(qualidade, 2)
    )

    st.metric(
        "Score de Qualidade dos Dados",
        f"{qualidade}%"
    )
    
    st.subheader("Insights da Qualidade")
    if qualidade >= 95:
        st.success("Excelente qualidade dos dados.")
    elif qualidade >= 85:
        st.info("Boa qualidade dos dados.")
    else:
        st.warning("A qualidade dos dados pode impactar os modelos.")