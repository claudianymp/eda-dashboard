import numpy as np
import pandas as pd
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def iniciar_etapa_pre_processamento(df_train, df_test):
    df_train = df_train.copy()
    df_test = df_test.copy()
    
    cols_targets = ['cancer_presence', 'cancer_subtype']
    
    tratar_variaveis_ordinais(df_train, df_test)
    
    df_train, df_test = tratar_variaveis_categoricas(df_train, df_test)

    features_para_processar = tratar_features_numericas(df_train, cols_targets)

    imputar_features(df_train, df_test, features_para_processar)

    X_train_scaled, X_test_scaled = normalizar_features(df_train, df_test, features_para_processar)

    st.success("Pré-processamento concluído com sucesso.")

    return (df_train,df_test,X_train_scaled,X_test_scaled)

def tratar_variaveis_ordinais(df_train, df_test):
    stage_mapping = {
        'no cancer': 0,
        'stage i': 1,
        'stage ii': 2,
        'stage iii': 3,
        'stage iv': 4,
        'stage v': 5
    }

    if 'cancer_stage' in df_train.columns:
        df_train['cancer_stage'] = (
            df_train['cancer_stage']
            .astype(str)
            .str.strip()
            .str.lower()
            .map(stage_mapping)
            .astype(float)
        )

        df_test['cancer_stage'] = (
            df_test['cancer_stage']
            .astype(str)
            .str.strip()
            .str.lower()
            .map(stage_mapping)
            .astype(float)
        )

    return df_train, df_test

def tratar_variaveis_categoricas(df_train, df_test):
    categorical_cols = [
        'smoking_history',
        'tumor_location',
        'patient_gender'
    ]

    categorical_cols = [
        c for c in categorical_cols
        if c in df_train.columns
    ]

    if categorical_cols:
        df_train = pd.get_dummies(
            df_train,
            columns=categorical_cols,
            drop_first=True
        )

        df_test = pd.get_dummies(
            df_test,
            columns=categorical_cols,
            drop_first=True
        )

    df_train, df_test = df_train.align(
        df_test,
        join='left',
        axis=1,
        fill_value=0
    )
    return df_train, df_test

def tratar_features_numericas(df_train, cols_targets):
    cols_numericas = df_train.select_dtypes(
        include=[np.number]
    ).columns

    features_para_processar = [
        c for c in cols_numericas
        if c not in cols_targets
    ]
    
    return features_para_processar

def imputar_features(df_train, df_test, features_para_processar):
    imputer = SimpleImputer(
        strategy='median'
    )

    df_train[features_para_processar] = (
        imputer.fit_transform(
            df_train[features_para_processar]
        )
    )

    df_test[features_para_processar] = (
        imputer.transform(
            df_test[features_para_processar]
        )
    )

def normalizar_features(df_train, df_test, features_para_processar):
    scaler = StandardScaler()

    df_train[features_para_processar] = (
        scaler.fit_transform(
            df_train[features_para_processar]
        )
    )

    df_test[features_para_processar] = (
        scaler.transform(
            df_test[features_para_processar]
        )
    )

    X_train_scaled = (
        df_train[features_para_processar]
        .values
    )

    X_test_scaled = (
        df_test[features_para_processar]
        .values
    )
    
    return X_train_scaled,X_test_scaled

