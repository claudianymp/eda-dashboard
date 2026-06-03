import plotly.express as px
import streamlit as st

def montar_visao_geral(df):
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total de Pacientes",len(df))
    col2.metric("Casos com Câncer",df['cancer_presence'].sum())
    col3.metric("Sem Câncer",len(df) - df['cancer_presence'].sum())
    col4.metric("Idade Média",round(df['patient_age'].mean(), 1))
    
def plot_distribuicao_geral_diagnostico(df):
    distribuicao = (
        df['cancer_presence']
        .value_counts()
        .reset_index()
    )
    distribuicao.columns = ['Diagnóstico', 'Quantidade']
    distribuicao['Diagnóstico'] = (
        distribuicao['Diagnóstico']
        .map({
            0: 'Câncer Não Detectado',
            1: 'Câncer Detectado'
        })
    )

    cores = {
        'Câncer Detectado': "#281ABA", 
        'Câncer Não Detectado': "#0564AC"      
    }

    fig = px.bar(
        distribuicao,
        x='Quantidade',
        y='Diagnóstico',
        color='Diagnóstico',
        color_discrete_map=cores,
        orientation='h',
        text='Quantidade',
        title='Distribuição Geral dos Diagnósticos'
    )
    fig.update_layout(
        xaxis_title='Quantidade de Pacientes',
        yaxis_title='',
        showlegend=False,
        height=350
    )
    st.plotly_chart(fig,use_container_width=True)
    
    total = len(df)
    positivos = (
        df['cancer_presence']
        .eq(1)
        .sum()
    )
    percentual = positivos / total * 100

    st.info(
        f"O dataset possui {total:,} pacientes, "
        f"dos quais {positivos:,} ({percentual:.1f}%) "
        f"apresentam diagnóstico positivo para câncer."
    )

