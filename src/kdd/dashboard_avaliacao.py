import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    silhouette_score
)

def dashboard_avaliacao_modelo(
    pipe_pca=None,
    X_test_pca=None,
    y_true=None,
    preds=None,
    probs=None,
    dbscan_labels=None
):
    st.header("Avaliação e Interpretação dos Resultados")
    
    if pipe_pca is None:
        st.warning("Pipeline PCA não encontrado.")
        return

    preds = preds.flatten()
    probs = probs.flatten()
    y_true = pd.Series(y_true).astype(int)

    plot_analise_loading_pca(pipe_pca)
    plot_densidade_separacao_classes_pca(X_test_pca, y_true)
    exibir_metricas(X_test_pca, y_true, preds, dbscan_labels)
    plot_matrix_confusao(y_true, preds)
    plot_curva_ROC(y_true, probs)
    plot_distribuicao_probabilidades(y_true, preds, probs)

def plot_analise_loading_pca(pipe_pca):
    st.subheader("Análise de Loadings (PCA)")
    
    pca = pipe_pca.named_steps['pca']
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[
            f'PC{i+1}'
            for i in range(
                pca.n_components_
            )
        ]
    )

    fig_loadings = px.imshow(
        loadings,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        title="Heatmap — Loadings PCA"
    )

    st.plotly_chart(fig_loadings,use_container_width=True)

def plot_densidade_separacao_classes_pca(X_test_pca, y_true):
    st.subheader("Densidade de Separação das Classes")
    
    separacao_df = X_test_pca.copy()
    separacao_df['Classe'] = y_true.values
    
    fig_sep = px.scatter(
        separacao_df,
        x='PC1',
        y='PC2',
        color='Classe',
        opacity=0.7,
        title="Separação das Classes no Espaço PCA"
    )

    st.plotly_chart(fig_sep,use_container_width=True)

def exibir_metricas(X_test_pca, y_true, preds, dbscan_labels):
    accuracy = accuracy_score(
        y_true,
        preds
    )
    precision = precision_score(
        y_true,
        preds,
        zero_division=0
    )
    recall = recall_score(
        y_true,
        preds,
        zero_division=0
    )
    f1 = f1_score(
        y_true,
        preds,
        zero_division=0
    )
    
    silhouette = None
    if dbscan_labels is not None:
        try:
            silhouette = silhouette_score(
                X_test_pca,
                dbscan_labels
            )

        except:
            silhouette = None

    st.subheader("Métricas de Performance")

    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric(
        "Accuracy",
        f"{accuracy:.3f}"
    )
    col2.metric(
        "Precision",
        f"{precision:.3f}"
    )
    col3.metric(
        "Recall",
        f"{recall:.3f}"
    )
    col4.metric(
        "F1-score",
        f"{f1:.3f}"
    )
    col5.metric(
        "Silhouette",
        "N/A"
        if silhouette is None
        else f"{silhouette:.3f}"
    )

def plot_matrix_confusao(y_true, preds):
    st.subheader("Matriz de Confusão")

    conf_matrix = confusion_matrix(
        y_true,
        preds
    )

    fig_cm = px.imshow(
        conf_matrix,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="Predito",y="Real"),
        title="Confusion Matrix"
    )

    st.plotly_chart(fig_cm,use_container_width=True)

def plot_curva_ROC(y_true, probs):
    st.subheader("Curva ROC")

    fpr,tpr,_ = roc_curve(
        y_true,
        probs
    )
    auc = roc_auc_score(
        y_true,
        probs
    )
    
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'AUC={auc:.3f}'
        )
    )

    fig_roc.add_trace(
        go.Scatter(
            x=[0,1],
            y=[0,1],
            mode='lines',
            line=dict(dash='dash'),
            name='Random'
        )
    )

    fig_roc.update_layout(
        title="Curva ROC",
        xaxis_title="Taxa Falso Positivo",
        yaxis_title="Taxa Verdadeito Positivo"
    )

    st.plotly_chart(fig_roc,use_container_width=True)

def plot_distribuicao_probabilidades(y_true, preds, probs):
    st.subheader("Resultados das Predições")

    resultado_df = pd.DataFrame({
        'Real': y_true,
        'Predito': preds,
        'Probabilidade':
        probs.round(4)
    })
    st.dataframe(resultado_df.head(50))

    fig_prob = px.histogram(
        resultado_df,
        x='Probabilidade',
        color='Real',
        nbins=30,
        marginal='box',
        title="Distribuição das Probabilidades"
    )

    st.plotly_chart(fig_prob,use_container_width=True)

  