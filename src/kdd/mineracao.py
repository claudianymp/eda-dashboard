import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from sklearn.cluster import DBSCAN
from torch.utils.data import TensorDataset, DataLoader
class LungCancerMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

def mineracao_dbscan(df_pca, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(df_pca)

    df_pca['Cluster'] = clusters
    
    n_ruidos = list(clusters).count(-1)
    st.warning(f"DBSCAN: Encontrados {len(set(clusters)) - (1 if -1 in clusters else 0)} clusters e {n_ruidos} pontos de ruído.")
    
    return df_pca

def mineracao_rede_neural(
    X_train,
    y_train,
    X_test=None,
    batch_size=32,
    learning_rate=0.001
):
    """
    Treina uma MLP para classificação binária.
    """
    st.write("X_train shape:", X_train.shape)
    if hasattr(y_train, "head"):
        st.write("Primeiras linhas do target:")
        st.write(y_train.head())
        st.write("1 = Tumor Maligno detectado, 0 = Sem cancer")

    if X_test is not None:
        st.write("X_test shape:", X_test.shape)

        X_test = X_test.reindex(
            columns=X_train.columns,
            fill_value=0
        )

        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(
                f"Incompatibilidade detectada: "
                f"X_train={X_train.shape[1]} features "
                f"X_test={X_test.shape[1]} features"
            )    
        
    if isinstance(y_train, pd.DataFrame):
        if y_train.shape[1] > 1:
            raise ValueError(
                "y_train deve conter apenas uma coluna target."
            )

        y_train = y_train.iloc[:, 0]

    y_train = y_train.astype('float32')
    X_tensor = torch.tensor(
        X_train.values,
        dtype=torch.float32
    )
    y_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32
    ).reshape(-1, 1)

    X_test_tensor = None
    if X_test is not None:

        X_test_tensor = torch.tensor(
            X_test.values,
            dtype=torch.float32
        )

    epochs = st.slider(
        "Quantidade de Épocas",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        key="slider_epochs"
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = LungCancerMLP(
        X_train.shape[1]
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    progress_bar = st.progress(0)
    loss_placeholder = st.empty()

    loss_history = []

    # ========= TREINAMENTO =========
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(
                outputs,
                batch_y
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(
            avg_loss
        )
        
        progress_bar.progress(
            (epoch + 1) / epochs
        )

        if epoch % 10 == 0 or epoch == epochs - 1:
            loss_placeholder.write(
                f"Época {epoch+1}/{epochs} | Loss: {avg_loss:.4f}"
            )

    st.success(
        f"Treinamento concluído! Loss final: {loss_history[-1]:.4f}"
    )

    # ========= INFERENCIA =========
    model.eval()
    with torch.no_grad():
        if X_test_tensor is not None:
            logits = model(
                X_test_tensor
            )
            st.info(
                "Predições realizadas usando X_test."
            )
        else:
            logits = model(
                X_tensor
            )
            st.warning(
                "X_test não informado. "
                "Usando dados de treino."
            )
            
        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities > 0.5
        ).float()

    return (
        model,
        loss_history,
        predictions.numpy(),
        probabilities.numpy()
    )