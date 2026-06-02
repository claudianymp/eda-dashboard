import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def iniciar_etapa_pre_processamento(df_train, df_test):
    cols_targets = ['cancer_presence', 'cancer_subtype']
    cols_numericas = df_train.select_dtypes(include=[np.number]).columns
    features_para_processar = [c for c in cols_numericas if c not in cols_targets]

    imputer = SimpleImputer(strategy='median')
    
    df_train[features_para_processar] = imputer.fit_transform(df_train[features_para_processar])
    df_test[features_para_processar] = imputer.transform(df_test[features_para_processar])

    scaler = StandardScaler()
    
    df_train[features_para_processar] = scaler.fit_transform(df_train[features_para_processar])
    df_test[features_para_processar] = scaler.transform(df_test[features_para_processar])

    X_train_scaled = df_train[features_para_processar].values
    X_test_scaled = df_test[features_para_processar].values
    
    st.success("Pré-processamento concluído: Tratamento de valores nulos e Normalização.")
    
    return df_train, df_test, X_train_scaled, X_test_scaled