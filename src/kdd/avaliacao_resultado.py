import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

def avaliar_modelo_classificacao(
    y_true,
    preds,
    probs
):
    st.header(
        "Avaliação e Interpretação dos resultados"
    )
    y_true = pd.Series(
        y_true
    ).astype(int)

    preds = preds.flatten().astype(int)
    probs = probs.flatten()

    accuracy, precision, recall, f1 = obter_metricas_modelo(y_true, preds)
    plot_matrix_confusao(y_true, preds)
    curva_roc = plot_curva_ROC(y_true, probs)
    visualizar_resultado(y_true, preds, probs)

    return {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1":f1,
        "curva_roc":curva_roc
    }

def plot_matrix_confusao(y_true, preds):
    st.subheader(
        "Matriz de Confusão"
    )
    conf_matrix = confusion_matrix(
        y_true,
        preds
    )

    fig, ax = plt.subplots()
    im = ax.imshow(conf_matrix)
    ax.set_title(
        "Matriz de Confusão"
    )
    ax.set_xlabel(
        "Predito"
    )
    ax.set_ylabel(
        "Real"
    )
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(
                j,
                i,
                str(conf_matrix[i,j]),
                ha='center',
                va='center'
            )

    st.pyplot(fig)
    
def plot_curva_ROC(y_true, probs):
    st.subheader(
        "Curva ROC"
    )
    fpr, tpr, thresholds = roc_curve(
        y_true,
        probs
    )
    auc = roc_auc_score(
        y_true,
        probs
    )
    fig2, ax2 = plt.subplots()
    ax2.plot(
        fpr,
        tpr
    )
    ax2.plot(
        [0,1],
        [0,1]
    )
    ax2.set_title(
        f"Curva ROC (AUC={auc:.3f})"
    )
    ax2.set_xlabel(
        "Taxa de Falso Positivo"
    )
    ax2.set_ylabel(
        "Taxa de Verdadeiro Positivo"
    )
    st.pyplot(
        fig2
    )
    
    return auc

def visualizar_resultado(y_true, preds, probs):
    st.subheader(
        "Visualização dos Resultados"
    )

    resultado_df = pd.DataFrame({
        "Real": y_true.values,
        "Predito": preds,
        "Probabilidade": probs.round(4)
    })
    st.dataframe(
        resultado_df.head(20)
    )
    st.bar_chart(
        resultado_df[
            ['Real','Predito']
        ].head(50)
    )

def obter_metricas_modelo(y_true, preds):
    st.subheader(
        "Métricas do Modelo"
    )
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

    col1,col2,col3,col4 = st.columns(4)
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
    
    return accuracy,precision,recall,f1


