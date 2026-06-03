import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

def analisar_fator_risco(
    df,
    coluna,
    titulo,
    question
):
    st.subheader(question)

    distribuicao = (
        df[coluna]
        .value_counts()
        .reset_index()
    )

    distribuicao.columns = [
        'Categoria',
        'Quantidade'
    ]

    fig = px.bar(
        distribuicao,
        x='Categoria',
        y='Quantidade',
        text='Quantidade',
        color='Categoria',
        title=f'Distribuição de {titulo}'
    )
    st.plotly_chart(fig, use_container_width=True)

    tabela = (
        df.groupby([coluna, 'cancer_presence'])
        .size()
        .reset_index(name='total')
    )

    tabela['Diagnóstico'] = tabela['cancer_presence'].map({0: 'Câncer Não Detectado', 1: 'Câncer Detectado'})

    fig = px.bar(
        tabela,
        x=coluna,
        y='total',
        color='Diagnóstico',
        barmode='group',
        text='total',
        title=f'Câncer por {titulo}'
    )
    st.plotly_chart(fig, use_container_width=True)

    percentual = (
        df.groupby(coluna)
          ['cancer_presence']
          .mean()
          .mul(100)
          .round(1)
          .reset_index()
    )

    percentual.columns = [
        'Categoria',
        'Percentual'
    ]

    fig = px.bar(
        percentual,
        x='Categoria',
        y='Percentual',
        text='Percentual',
        title=f'Percentual de Câncer por {titulo}'
    )
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Resumo Percentual")
    st.dataframe(percentual, use_container_width=True)
    

def heatmap_fatores_risco(df):
    st.subheader("Comparação dos Fatores de Risco")
    
    fatores = [
        'smoking_history',
        'family_history',
    ]

    resultados = []
    for fator in fatores:
        medias = (
            df.groupby(fator)
            ['cancer_presence']
            .mean()
            .mul(100)
        )

        for categoria, valor in medias.items():
            resultados.append([
                fator,
                str(categoria),
                valor
            ])

    heatmap_df = pd.DataFrame(
        resultados,
        columns=[
            'Fator',
            'Categoria',
            'Percentual'
        ]
    )

    fig = px.density_heatmap(
        heatmap_df,
        x='Categoria',
        y='Fator',
        z='Percentual',
        text_auto='.1f'
    )
    st.plotly_chart(fig, use_container_width=True)    