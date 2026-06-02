import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def iniciar_etapa_transformacao_pca(df_train, df_test, n_componentes=3):
    cols_targets = ['cancer_presence', 'cancer_subtype']
    cols_numericas = df_train.select_dtypes(include=[np.number]).columns
    features_para_processar = [c for c in cols_numericas if c not in cols_targets]

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_componentes))
    ])

    X_train_pca_array = pipe.fit_transform(df_train[features_para_processar])
    X_test_pca_array = pipe.transform(df_test[features_para_processar])

    pca_object = pipe.named_steps['pca']
    variancia = pca_object.explained_variance_ratio_.sum() * 100
    
    cols_pca = [f'PC{i+1}' for i in range(n_componentes)]
    df_train_pca = pd.DataFrame(data=X_train_pca_array, columns=cols_pca)
    df_test_pca = pd.DataFrame(data=X_test_pca_array, columns=cols_pca)
    
    st.info(f"PCA concluído: {n_componentes} componentes explicam {variancia:.2f}% da variância do conjunto de treino.")
    
    return df_train_pca, df_test_pca, pipe

def visualizar_pca_3d(df_pca, df_original, coluna_color):
    df_plot = df_pca.copy()
    
    if coluna_color == 'Diagnóstico':
        df_plot['Color_Column'] = df_original['cancer_presence'].values
    else:
        df_plot['Color_Column'] = df_pca[coluna_color].values

    fig = px.scatter_3d(
        df_plot, x='PC1', y='PC2', z='PC3',
        color='Color_Column',
        title=f"Espaço PCA colorido por: {coluna_color}",
        color_continuous_scale='Viridis',
        opacity=0.7,
        height=600
    )
    
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)
    