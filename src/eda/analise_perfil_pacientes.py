import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

def graficos_cancer_por_genero(df):
    st.subheader("Como está distribuído o diagnóstico por gênero?")
    
    resumo = (
        df.groupby(['patient_gender', 'cancer_presence'])
        .size()
        .reset_index(name='total')
    )
    resumo['diagnostico'] = resumo['cancer_presence'].map({ 0: 'Câncer Não Detectado', 1: 'Câncer Detectado'})
    tabela = resumo.pivot_table(
        index='patient_gender',
        columns='diagnostico',
        values='total',
        fill_value=0
    )

    st.markdown("#### Tabela Resumo")
    st.dataframe(tabela, use_container_width=True)
    
    cores = {
        'Câncer Detectado': "#281ABA", 
        'Câncer Não Detectado': "#0564AC"      
    }

    st.markdown("#### Percentual de diagnóstico por Gênero") 
    total_homens = len(df[df['patient_gender'] == 'Male'])
    total_mulheres = len(df[df['patient_gender'] == 'Female'])
    
    cancer_homens = len(
        df[
            (df['patient_gender'] == 'Male')
            &
            (df['cancer_presence'] == 1)
        ]
    )
    cancer_mulheres = len(
        df[
            (df['patient_gender'] == 'Female')
            &
            (df['cancer_presence'] == 1)
        ]
    )

    perc_homens = (cancer_homens / total_homens) * 100
    perc_mulheres = (cancer_mulheres / total_mulheres) * 100
    percentuais = (
        df.groupby('patient_gender')['cancer_presence']
        .value_counts(normalize=True)
        .mul(100)
        .rename('percentual')
        .reset_index()
    )
    percentuais['diagnostico'] = percentuais['cancer_presence'].map({ 0: 'Câncer Não Detectado', 1: 'Câncer Detectado'})

    fig = px.bar(
        percentuais,
        x='patient_gender',
        y='percentual',
        color='diagnostico',
        color_discrete_map=cores,
        barmode='group',
        text='percentual',
    )
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        hovertemplate=
        '<b>%{x}</b><br>' +
        'Percentual: %{y:.1f}%<extra></extra>'
    )
    fig.update_layout(
        yaxis_title='Percentual (%)',
        xaxis_title='Gênero',
        legend_title='Diagnóstico'
    )
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("#### Insights")
    if perc_homens > perc_mulheres:
        st.success(
            f"Homens apresentam maior incidência de câncer "
            f"({perc_homens:.1f}%) em comparação às mulheres "
            f"({perc_mulheres:.1f}%)."
        )
    else:
        st.success(
            f"Mulheres apresentam maior incidência de câncer "
            f"({perc_mulheres:.1f}%) em comparação aos homens "
            f"({perc_homens:.1f}%)."
        )
        
    diff = abs(perc_homens - perc_mulheres)
    st.info(
        f"A diferença entre os gêneros é de apenas "
        f"{diff:.1f}% pontos percentuais, sugerindo "
        f"uma distribuição bastante semelhante dos diagnósticos."
    ) 
    
def graficos_cancer_por_idade(df):
    graficos_dist_frequencia(df)
    resumo_idade_por_diagnostico(df)
    faixa_etaria_diagnostico(df)

def graficos_dist_frequencia(df):
    st.subheader("Como a idade está distribuída entre pacientes com e sem câncer?")
    grafico_distribuicao_idade(df)
    boxplot_idade_por_diagnostico(df)
    distribuicao_frequencia_por_idade(df)

def distribuicao_frequencia_por_idade(df):
    st.write("Histograma por diagnóstico por idade")
    
    df_plot = df.copy()
    df_plot['diagnostico'] = df_plot['cancer_presence'].map({ 0: 'Câncer Não Detectado', 1: 'Câncer Detectado'})
    cores = {
        'Câncer Detectado': "#281ABA", 
        'Câncer Não Detectado': "#0564AC"      
    }

    fig = px.histogram(
        df_plot,
        x='patient_age',
        color='diagnostico',
        color_discrete_map=cores,
        barmode='overlay',
        nbins=20,
        opacity=0.7,
    )
    fig.update_layout(
        xaxis_title='Idade',
        yaxis_title='Quantidade de Pacientes',
        legend_title='Diagnóstico'
    )
    st.plotly_chart(fig, use_container_width=True)

def boxplot_idade_por_diagnostico(df):
    st.write("BoxPlot - Distribuição da Idade por Diagnóstico")
    df_plot = df.copy()
    df_plot['diagnostico'] = df_plot['cancer_presence'].map({ 0: 'Câncer Não Detectado', 1: 'Câncer Detectado'})
    
    cores = {
        'Câncer Detectado': "#5446EF", 
        'Câncer Não Detectado': "#33A1F5"      
    }
    
    fig = px.box(
        df_plot,
        x='diagnostico',
        y='patient_age',
        color='diagnostico',
        color_discrete_map=cores,
    )
    fig.update_layout(
        xaxis_title='Diagnóstico',
        yaxis_title='Idade',
        legend_title='Diagnóstico'
    )
    st.plotly_chart(fig, use_container_width=True)

def grafico_distribuicao_idade(df):
    st.write("Distribuição com Curvas de densidade separadas")

    idade_sem = df[df['cancer_presence'] == 0]['patient_age']
    idade_com = df[df['cancer_presence'] == 1]['patient_age']

    fig = ff.create_distplot(
        [idade_sem, idade_com],
        ['Não Detectado', 'Detectado'],
        show_hist=False,
        show_rug=False
    )

    fig.update_layout(
        xaxis_title='Idade',
        yaxis_title='Densidade'
    )

    st.plotly_chart(fig, use_container_width=True)

def resumo_idade_por_diagnostico(df):
    resumo = (
        df.groupby('cancer_presence')['patient_age']
        .agg([
            'count',
            'min',
            'max',
            'mean',
            'median',
            'std'
        ])
        .round(1)
    )

    resumo.index = [
        'Câncer Não Detectado',
        'Câncer Detectado'
    ]

    st.subheader("Resumo Estatístico")
    st.dataframe(resumo, use_container_width=True)

def faixa_etaria_diagnostico(df):
    st.subheader("Diagnóstico por Faixa Etária")
    bins = [
        0,
        40,
        50,
        60,
        70,
        80,
        120
    ]

    labels = [
        '<40',
        '40-49',
        '50-59',
        '60-69',
        '70-79',
        '80+'
    ]

    df_plot = df.copy()
    df_plot['faixa_etaria'] = pd.cut(
        df_plot['patient_age'],
        bins=bins,
        labels=labels
    )

    tabela = (
        df_plot.groupby(
            ['faixa_etaria', 'cancer_presence']
        )
        .size()
        .reset_index(name='total')
    )

    tabela['diagnostico'] = tabela['cancer_presence'].map({ 0: 'Câncer Não Detectado', 1: 'Câncer Detectado'})
    cores = {
        'Câncer Detectado': "#281ABA", 
        'Câncer Não Detectado': "#0564AC"      
    }
    fig = px.bar(
        tabela,
        x='faixa_etaria',
        y='total',
        color='diagnostico',
        color_discrete_map=cores,
        barmode='group',
        text='total',
    )
    fig.update_layout(
        legend_title='Diagnóstico'
    )
    st.plotly_chart(fig, use_container_width=True)

    media_cancer = df[df['cancer_presence'] == 1]['patient_age'].mean()
    media_sem_cancer = df[df['cancer_presence'] == 0]['patient_age'].mean()
    st.info(
        f"Pacientes com câncer apresentam idade média de "
        f"{media_cancer:.1f} anos, enquanto pacientes sem "
        f"câncer apresentam média de "
        f"{media_sem_cancer:.1f} anos."
    )

def graficos_cancer_por_estagio(df):
    grafico_distribuicao_estagios(df)
    grafico_estagio_por_genero(df) 
    percentual_estagio_por_genero(df) 
    grafico_idade_por_estagio(df)
    heatmap_estagio_genero(df)
    resumo_estagios(df)
    
    st.subheader("Insights")
    estagio_mais_comum = (
        df['cancer_stage']
        .value_counts()
        .idxmax()
    )

    percentual = (
        df['cancer_stage']
        .value_counts(normalize=True)
        .max()
        * 100
    )

    st.info(
        f"O estágio mais frequente é "
        f"'{estagio_mais_comum}', representando "
        f"{percentual:.1f}% dos registros."
    )

def grafico_distribuicao_estagios(df):
    st.subheader("Qual a distribuição dos pacientes por estágio do câncer?")
    
    distribuicao = (
        df['cancer_stage']
        .value_counts()
        .reset_index()
    )
    distribuicao.columns = [
        'Estágio',
        'Quantidade'
    ]
    distribuicao['Percentual'] = (
        distribuicao['Quantidade']
        / distribuicao['Quantidade'].sum()
        * 100
    ).round(1)

    ordem = [
        'No Cancer',
        'Stage I',
        'Stage II',
        'Stage III',
        'Stage IV',
        'Stage V'
    ]
    distribuicao['Estágio'] = pd.Categorical(
        distribuicao['Estágio'],
        categories=ordem,
        ordered=True
    )
    distribuicao = distribuicao.sort_values('Estágio')

    fig = px.bar(
        distribuicao,
        x='Quantidade',
        y='Estágio',
        orientation='h',
        text='Percentual'
    )
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )
    fig.update_layout(
        xaxis_title='Quantidade de Pacientes',
        yaxis_title=''
    )

    st.plotly_chart(fig, use_container_width=True)
    
def grafico_estagio_por_genero(df):
    st.subheader("Homens e mulheres apresentam distribuição semelhante dos estágios?")
    
    tabela = (
        df.groupby(
            ['cancer_stage', 'patient_gender']
        )
        .size()
        .reset_index(name='total')
    )

    fig = px.bar(
        tabela,
        x='cancer_stage',
        y='total',
        color='patient_gender',
        barmode='group',
        text='total',
    )
    fig.update_layout(
        xaxis_title='Estágio',
        yaxis_title='Quantidade'
    )

    st.plotly_chart(fig, use_container_width=True )

def percentual_estagio_por_genero(df):
    st.subheader("Distribuição Percentual dos Estágios por Gênero")

    tabela = (
        df.groupby(
            ['patient_gender']
        )['cancer_stage']
        .value_counts(normalize=True)
        .mul(100)
        .rename('percentual')
        .reset_index()
    )

    fig = px.bar(
        tabela,
        x='patient_gender',
        y='percentual',
        color='cancer_stage',
        barmode='stack',
        text='percentual',
    )
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='inside'
    )

    st.plotly_chart(fig, use_container_width=True)    
    
def grafico_idade_por_estagio(df):
    st.subheader("Pacientes em estágios mais avançados tendem a ser mais velhos?")

    fig = px.box(
        df,
        x='cancer_stage',
        y='patient_age',
        color='cancer_stage',
    )

    fig.update_layout(
        xaxis_title='Estágio',
        yaxis_title='Idade'
    )

    st.plotly_chart(fig, use_container_width=True)    
    
def heatmap_estagio_genero(df):
    st.subheader("Heatmap: Estágio do Câncer por Gênero")
    
    tabela = pd.crosstab(
        df['cancer_stage'],
        df['patient_gender']
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=tabela.values,
            x=tabela.columns,
            y=tabela.index,
            text=tabela.values,
            texttemplate="%{text}",
            hoverongaps=False
        )
    )
    st.plotly_chart(fig, use_container_width=True) 
 
def resumo_estagios(df):
    st.subheader("Resumo dos estágios")
    
    resumo = (
        df['cancer_stage']
        .value_counts()
        .reset_index()
    )

    resumo.columns = [
        'Estágio',
        'Quantidade'
    ]

    resumo['Percentual'] = (
        resumo['Quantidade']
        / resumo['Quantidade'].sum()
        * 100
    ).round(2)

    st.dataframe(resumo, use_container_width=True)
 
        