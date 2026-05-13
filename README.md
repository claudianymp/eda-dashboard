Este projeto é um artefato técnico desenvolvido para a disciplina de Projeto Interdisciplinar para Sistemas de Informação 3 (PISI3) do 3° período do curso de Bacharelado em Sistemas de Informação (BSI) e do Programa de Pós-Graduação em Informática Aplicada da Universidade Federal Rural de Pernambuco (UFRPE).

por: Claudiany Martins (claudiany.martins@ufrpe.br)


##  Dataset para Classificação de Câncer de Pulmão

### Dataset information
[kaggle - Lung Cancer Dataset ](https://www.kaggle.com/datasets/datasetengineer/lungcanc2024?utm_source=chatgpt.com&select=LungCanC2024_Dataset.csv)
 
O dataset **LungCanC2024** oferece uma infraestrutura de dados em larga escala para o suporte à decisão clínica no diagnóstico do câncer de pulmão. Composto por 289,010 mil registros, o conjunto unifica dados de alta complexidade, incluindo biomarcadores moleculares, características radiômicas extraídas de exames de imagem e metadados clínico-demográficos. O diferencial deste repositório reside na sua natureza multirrótulo, permitindo que algoritmos sejam treinados simultaneamente para a identificação de presença de câncer, subtipagem histológica, estadiamento (I-IV) e predição de tempo de sobrevida, sendo um recurso essencial para experimentação em redes neurais profundas e arquiteturas de aprendizado distribuído.

## Predição e Análise de Câncer de Pulmão

Este projeto aplica técnicas de **KDD (Knowledge Discovery in Databases)** e **Deep Learning** para a detecção e classificação de tumores pulmonares, integrando dados radiómicos, clínicos e genómicos.

---

### Descrição das Funcionalidades (Dataset)

O conjunto de dados está estruturado em três categorias principais, permitindo uma análise multifatorial do paciente.

### 1. Características de Imagem (Radiómica)
Dados extraídos de exames de imagem como CT (Tomografia) e PET Scans.

| Nome do Atributo   | Descrição                                                                      |
| :----------------- | :----------------------------------------------------------------------------- |
| `nodule_size_mm`   | Tamanho dos nódulos pulmonares detectados (em mm).                             |
| `nodule_texture`   | Característica de textura derivada da análise radiológica.                     |
| `HU_mean`          | Valor médio da Unidade Hounsfield (HU) em exames de CT.                        |
| `HU_std`           | Desvio padrão dos valores de HU (indica variação de densidade).                |
| `GLCM_contrast`    | Contraste da matriz de co-ocorrência de níveis de cinza (heterogeneidade).     |
| `GLCM_correlation` | Métrica de correlação GLCM que avalia a consistência do padrão.                |
| `PET_SUVmax`       | Valor máximo de captação padronizada (SUV) do PET scan (atividade metabólica). |
| `PET_SUVmean`      | Valor médio de SUV em toda a região do tumor.                                  |

### 2. Características Clínicas e Metadados
Informações sobre o perfil do paciente e histórico de saúde.

| Nome do Atributo            | Descrição                                                     |
| :-------------------------- | :------------------------------------------------------------ |
| `patient_age`               | Idade do paciente (intervalo de 30 a 90 anos).                |
| `patient_gender`            | Género: Masculino (70%) / Feminino (30%).                     |
| `smoking_history`           | Histórico tabágico: Nunca fumou, Ex-fumante ou Fumante atual. |
| `family_history`            | Histórico familiar de câncer de pulmão (1 = Sim, 0 = Não).    |
| `tumor_location`            | Localização: Pulmão Esquerdo (40%) ou Direito (60%).          |
| `tumor_stage`               | Classificação do estágio do tumor (I a IV).                   |
| `radiation_therapy`         | Se o paciente recebeu radioterapia (Binário).                 |
| `chemotherapy_received`     | Se o paciente recebeu quimioterapia (Binário).                |
| `immunotherapy_received`    | Se o paciente recebeu imunoterapia (Binário).                 |
| `targeted_therapy_received` | Se o paciente recebeu terapia-alvo (Binário).                 |

### 3. Características Genómicas e Biomarcadores
Marcadores moleculares que indicam mutações específicas e instabilidade genómica.

| Nome do Atributo          | Descrição                                                             |
| :------------------------ | :-------------------------------------------------------------------- |
| `EGFR_mutation_status`    | Detecção de mutação no gene EGFR (Binário).                           |
| `KRAS_mutation_status`    | Detecção de mutação no gene KRAS (Binário).                           |
| `ALK_fusion_status`       | Presença de fusão do gene ALK (Binário).                              |
| `PD-L1_expression_level`  | Nível de expressão do biomarcador PD-L1 (0-100%).                     |
| `tumor_mutational_burden` | Carga Mutacional do Tumor (TMB), indicador de instabilidade genómica. |

---

### Variáveis Alvo (Multi-Label)

O modelo é treinado para prever múltiplos outputs simultaneamente:

* **`cancer_presence`**: Identificação binária (1 = Tumor Maligno, 0 = Sem Câncer).
* **`cancer_subtype`**: Classificação do tipo (Adenocarcinoma, Carcinoma Escamoso, Células Pequenas, Outros).
* **`cancer_stage`**: Estágio da progressão (Estágio I ao IV).
* **`survival_time_months`**: Estimativa de tempo de sobrevivência em meses.

---

### Tecnologias e Metodologia

Este projeto utiliza um pipeline de **Data Science** robusto:
1.  **EDA (Exploratory Data Analysis)**: Visualização de separação de classes e detecção de outliers.
2.  **Redução de Dimensionalidade**: Uso de **PCA** e **t-SNE** para entender a variância dos dados.
3.  **Modelagem**: Rede Neural **MLP (Multi-Layer Perceptron)** construída em **PyTorch**.
4.  **Interface**: Dashboard interativo desenvolvido com **Streamlit**.

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
