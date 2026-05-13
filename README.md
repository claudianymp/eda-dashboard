Este projeto é um artefato técnico desenvolvido para a disciplina de Projeto Interdisciplinar para Sistemas de Informação 3 (PISI3) do 3° período do curso de Bacharelado em Sistemas de Informação (BSI) e do Programa de Pós-Graduação em Informática Aplicada da Universidade Federal Rural de Pernambuco (UFRPE).

por: Claudiany Martins (claudiany.martins@ufrpe.br)


##  Dataset para Classificação de Câncer de Pulmão


 
### LungCanC2024: Predição Multimodal e Clustering de Câncer de Pulmão

Este repositório apresenta o desenvolvimento de um ecossistema de análise preditiva para o cancer do pulmão, utilizando o dataset [Lung Cancer Dataset ](https://www.kaggle.com/datasets/datasetengineer/lungcanc2024?utm_source=chatgpt.com&select=LungCanC2024_Dataset.csv). O projeto integra técnicas de **Redes Neuronais (MLP)** para classificação e **DBSCAN** para a descoberta de padrões não supervisionados, seguindo a metodologia rigorosa de **KDD (Knowledge Discovery in Databases)**.

---

### Descrição do Projeto

O cancer do pulmão é uma das principais causas de mortalidade oncológica global. Este projeto visa fornecer modelos diagnósticos precisos através da integração de dados multidimensionais: **Radiómica (imagem), Clínica e Genómica**.

O dataset **LungCanC2024** contém **289.010 registos**, permitindo pesquisas avançadas em medicina de precisão, diagnóstico assistido por computador e análise de sobrevivência.

---

### Estrutura do Dataset

Os dados estão organizados em quatro pilares fundamentais para garantir uma visão holística do paciente:

#### 1. Características Radiómicas (Imagem)
Métricas extraídas de Tomografias Computadorizadas (CT) e PET Scans.
* `nodule_size_mm`: Diâmetro dos nódulos.
* `HU_mean / HU_std`: Densidade tecidual em Unidades Hounsfield.
* `GLCM_contrast / Correlation`: Textura e heterogeneidade do tumor.
* `PET_SUVmax / SUVmean`: Atividade metabólica tumoral.

#### 2. Metadados Clínicos
* **Demografia**: Idade (30-90 anos) e Género.
* **Histórico**: Tabagismo e histórico familiar.
* **Tratamentos**: Registos de Radioterapia, Quimioterapia, Imunoterapia e Terapia Alvo.

#### 3. Biomarcadores Genómicos
* Status de mutação: **EGFR**, **KRAS** e fusão **ALK**.
* Expressão de **PD-L1** e Carga Mutacional do Tumor (**TMB**).

#### 4. Variáveis Alvo (Multi-Label)
* `cancer_presence`: Diagnóstico binário (Maligno/Benigno).
* `cancer_subtype`: Subtipagem histológica.
* `cancer_stage`: Estadiamento clínico (I a IV).
* `survival_time_months`: Predição de sobrevivência.

---

### Metodologia e Algoritmos

O pipeline de inteligência artificial foi estruturado em duas frentes principais:

#### Aprendizado Supervisionado (Classificação)
Implementação de uma **Rede Neuronal MLP (Multi-Layer Perceptron)** via PyTorch para predição multirrótulo, focada na alta acurácia diagnóstica.

#### Aprendizado Não Supervisionado (Clustering)
Uso do algoritmo **DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*) para:
* **Deteção de Anomalias**: Identificação automática de outliers e perfis clínicos raros (ruído).
* **Validação de Grupos**: Verificação de como as variáveis radiómicas e genómicas se agrupam naturalmente sem a influência dos rótulos pré-existentes.
* **Análise de Densidade**: Diferenciação de subtipos tumorais baseada na densidade de suas características biológicas.

---

### Fluxo KDD (Processo de Descoberta)

1.  **Seleção e Integração**: Consolidação da base LungCanC2024.
2.  **Pré-processamento**: Tratamento de valores omissos e normalização robusta (**StandardScaler**).
3.  **Transformação**: Redução de dimensionalidade via **PCA** (Principal Component Analysis) para otimizar o desempenho do DBSCAN e da MLP.
4.  **Mineração de Dados**: Execução do treino da Rede Neuronal e do agrupamento espacial.
5.  **Interpretação/Avaliação**: Dashboard interativo para análise de loadings, densidades de separação e métricas de performance (Acurácia, F1-Score e Coeficiente de Silhueta).

---

### Tecnologias Utilizadas

* **Python 3.10+**
* **Streamlit**: Interface do Dashboard.
* **PyTorch**: Framework de Deep Learning.
* **Scikit-Learn**: Algoritmos de ML e Pré-processamento.
* **Plotly**: Visualização avançada de dados.

---

### Instalação

1. Instalar as dependencias

   ```
   $ pip install -r requirements.txt
   ```

2. Executar o app

   ```
   $ streamlit run streamlit_app.py
   ```
