import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def distribuicao_variavel(
    df,
    coluna,
    titulo
):
    st.subheader(titulo)
    fig = px.histogram(
        df,
        x=coluna,
        color="cancer_presence",
        nbins=30,
        marginal="box",
        opacity=0.7,
        barmode="overlay",
    )

    fig.update_layout(
        xaxis_title=coluna,
        yaxis_title="Quantidade"
    )
    st.plotly_chart(fig, use_container_width=True)
    
def boxplot_diagnostico(
    df,
    coluna,
):
    fig = px.box(
        df,
        x="cancer_presence",
        y=coluna,
        color="cancer_presence",
    )

    fig.update_layout(
        xaxis_title="Diagnóstico",
        yaxis_title=coluna
    )
    st.plotly_chart(fig, use_container_width=True)

def resumo_estatistico(df):
    st.subheader("Resumo Estatístico")

    colunas = [
        'patient_age',
        'nodule_size_mm',
        'PET_SUVmax',
        'HU_mean',
        'HU_std'
    ]

    st.dataframe(
        df.groupby(
            'cancer_presence'
        )[colunas]
        .agg([
            'mean',
            'median',
            'std'
        ])
    )   
    
def correlacao_variaveis_numericas(df):
    cols_num = [
        'patient_age',
        'nodule_size_mm',
        'PET_SUVmax',
        'HU_mean',
        'HU_std'
    ]

    corr = df[cols_num].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
    )
    st.plotly_chart(fig, use_container_width=True)    
    
    