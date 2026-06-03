import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

def distribuicao_geral_casos(df):
    st.subheader("Qual a proporção de pacientes com e sem câncer?")

    tabela = (
        df['cancer_presence']
        .value_counts()
        .reset_index()
    )

    tabela.columns = [
        'Diagnóstico',
        'Quantidade'
    ]

    tabela['Diagnóstico'] = tabela['Diagnóstico'].map({ 0: 'Câncer Não Detectado', 1: 'Câncer Detectado'})
    tabela['Percentual'] = (
        tabela['Quantidade']
        / tabela['Quantidade'].sum()
        * 100
    )

    fig = px.bar(
        tabela,
        x='Quantidade',
        y='Diagnóstico',
        orientation='h',
        color='Diagnóstico',
        text='Percentual',
    )

    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )

    st.plotly_chart(fig, use_container_width=True)

def distribuicao_estagios(df):
    st.subheader("Em quais estágios os pacientes são diagnosticados?")

    df_plot = df[df['cancer_stage'] != 'No Cancer']

    tabela = (
        df_plot['cancer_stage']
        .value_counts()
        .reset_index()
    )

    tabela.columns = [
        'Estágio',
        'Quantidade'
    ]

    fig = px.bar(
        tabela,
        x='Estágio',
        y='Quantidade',
        color='Estágio',
        text='Quantidade',
    )

    st.plotly_chart(fig, use_container_width=True)    
    
def distribuicao_subtipos(df):
    st.subheader("Qual subtipo é mais frequente?")

    df_plot = df[df['cancer_presence'] == 1]
    tabela = (
        df_plot['cancer_subtype']
        .value_counts()
        .reset_index()
    )

    tabela.columns = [
        'Subtipo',
        'Quantidade'
    ]

    tabela['Percentual'] = (
        tabela['Quantidade']
        / tabela['Quantidade'].sum()
        * 100
    )

    fig = px.bar(
        tabela,
        x='Subtipo',
        y='Quantidade',
        color='Subtipo',
        text='Percentual',
    )

    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )

    st.plotly_chart(fig,use_container_width=True)    
    
def progressao_estagios(df):
    st.subheader("Progressão dos Casos por Estágio")
    
    df_plot = df[df['cancer_stage'] != 'No Cancer']
    tabela = (
        df_plot['cancer_stage']
        .value_counts()
        .reset_index()
    )

    tabela.columns = [
        'Estágio',
        'Quantidade'
    ]

    ordem = [
        'Stage I',
        'Stage II',
        'Stage III',
        'Stage IV',
        'Stage V'
    ]

    tabela['Estágio'] = pd.Categorical(
        tabela['Estágio'],
        categories=ordem,
        ordered=True
    )

    tabela = tabela.sort_values('Estágio')
    fig = px.line(
        tabela,
        x='Estágio',
        y='Quantidade',
        markers=True,
    )
    st.plotly_chart(fig, use_container_width=True)
    
def heatmap_estagio_subtipo(df):
    st.subheader("Estágio x Subtipo")

    df_plot = df[df['cancer_presence'] == 1]
    tabela = pd.crosstab(
        df_plot['cancer_stage'],
        df_plot['cancer_subtype']
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=tabela.values,
            x=tabela.columns,
            y=tabela.index,
            text=tabela.values,
            texttemplate="%{text}"
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
def indicador_metricas(df):    
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Casos Detectados",
        df['cancer_presence'].sum()
    )

    col2.metric(
        "Subtipos",
        df['cancer_subtype'].nunique()
    )

    col3.metric(
        "Estágios",
        df['cancer_stage'].nunique() - 1
    )

    col4.metric(
        "Estágio Mais Frequente",
        df[df['cancer_stage'] != 'No Cancer']['cancer_stage']
        .mode()[0]
    )
    