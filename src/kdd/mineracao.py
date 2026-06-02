import streamlit as st
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn

class LungCancerMLP(nn.Module):
    def __init__(self, input_dim):
        super(LungCancerMLP, self).__init__()
        self.camadas = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.camadas(x)

def mineracao_rede_neural(X_train, y_train):
    X_tensor = torch.FloatTensor(X_train.values)
    y_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    
    epochs = st.slider("Quantidade de Épocas:", 10, 500, 100, key="slider_epochs")
    
    model = LungCancerMLP(X_train.shape[1])
    criterion = nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
    st.success("Treinamento da MLP concluído com sucesso!")
    return model

def mineracao_dbscan(df_pca, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(df_pca)
    
    df_pca['Cluster'] = clusters
    
    n_ruidos = list(clusters).count(-1)
    st.warning(f"DBSCAN: Encontrados {len(set(clusters)) - (1 if -1 in clusters else 0)} clusters e {n_ruidos} pontos de ruído.")
    
    return df_pca