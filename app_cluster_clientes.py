import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Configurações de estilo do Seaborn
sns.set(style="whitegrid")

# Carregar os modelos de clusterização
modelos = {
    "kmeans1": joblib.load("modelos/kmeans1.joblib"),
    "kmeans2": joblib.load("modelos/kmeans2.joblib"),
    "dbscan1": joblib.load("modelos/dbscan1.joblib"),
    "dbscan2": joblib.load("modelos/dbscan2.joblib"),
    "hierarchical1": joblib.load("modelos/hierarchical1.joblib"),
    "hierarchical2": joblib.load("modelos/hierarchical2.joblib")
}
scaler = RobustScaler()  # Ajuste para o scaler usado durante o treinamento

# Função para carregar a base de dados
@st.cache_data
def carregar_dados():
    return pd.read_excel("Base_Missão_S.xlsx", engine="openpyxl")

# Carregar a base de dados
dados_brutos = carregar_dados()

# Filtro por Segmento
st.sidebar.write("### Filtro por Segmento")
segmentos_disponiveis = dados_brutos['Segmento'].unique()
segmento_selecionado = st.sidebar.selectbox("Escolha o segmento para análise", segmentos_disponiveis)
dados_filtrados = dados_brutos[dados_brutos['Segmento'] == segmento_selecionado]

# Variáveis numéricas e categóricas para análise
numerical_features = ['Volume transacional (R$)', 'Receita Transacional (R$)', 'Receita Antecipação de recebíveis (R$)', 'CAC (R$)_x']
categorical_feature = 'Segmento'

# Função para calcular métricas de avaliação
def evaluate_clustering(labels, data):
    silhouette = silhouette_score(data, labels) if len(set(labels)) > 1 else None
    calinski_harabasz = calinski_harabasz_score(data, labels) if len(set(labels)) > 1 else None
    davies_bouldin = davies_bouldin_score(data, labels) if len(set(labels)) > 1 else None
    return silhouette, calinski_harabasz, davies_bouldin

# Calcular métricas para cada modelo e cada segmento
st.write("### Tabela de Métricas de Avaliação dos Modelos de Clusterização")
metricas = []

for modelo_nome, modelo in modelos.items():
    # Escalar os dados e aplicar o modelo
    dados_segmento_escalados = scaler.fit_transform(dados_filtrados[numerical_features])
    if "dbscan" in modelo_nome:
        labels = modelo.fit_predict(dados_segmento_escalados)
    else:
        labels = modelo.predict(dados_segmento_escalados)
    
    # Calcular as métricas de avaliação
    silhouette, calinski_harabasz, davies_bouldin = evaluate_clustering(labels, dados_segmento_escalados)
    metricas.append({
        'Modelo': modelo_nome,
        'Silhouette Score': silhouette,
        'Calinski-Harabasz Index': calinski_harabasz,
        'Davies-Bouldin Index': davies_bouldin
    })

# Criar DataFrame com os resultados das métricas
metricas_df = pd.DataFrame(metricas)
st.write(metricas_df)

# EDA - Visualização de Distribuição das Variáveis Numéricas
st.write(f"### Distribuição das Variáveis Numéricas - Segmento: {segmento_selecionado}")
for col in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(dados_filtrados[col], kde=True)
    plt.title(f'Distribuição de {col} - Segmento: {segmento_selecionado}')
    st.pyplot(plt)
    plt.clf()  # Limpa o gráfico após cada exibição para evitar sobreposição

# EDA - Visualização das Variáveis Categóricas
st.write(f"### Distribuição da Variável Categórica - Segmento: {segmento_selecionado}")
plt.figure(figsize=(8, 4))
sns.countplot(x=dados_filtrados[categorical_feature])
plt.title(f'Distribuição de {categorical_feature} - Segmento: {segmento_selecionado}')
plt.xticks(rotation=45)
st.pyplot(plt)
plt.clf()

# EDA - Matriz de Correlação das Variáveis Numéricas
numeric_df = dados_filtrados.select_dtypes(include=['number'])  # Seleciona apenas colunas numéricas
if not numeric_df.empty:
    st.write(f"### Matriz de Correlação das Variáveis Numéricas - Segmento: {segmento_selecionado}")
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f'Matriz de Correlação - Segmento: {segmento_selecionado}')
    st.pyplot(plt)
else:
    st.write("Não há colunas numéricas para calcular a matriz de correlação.")

# Box Plot Unificado para Variáveis Numéricas por Segmento
st.write(f"### Box Plot de Variáveis Numéricas por Segmento")
melted_df = pd.melt(dados_brutos, id_vars=[categorical_feature], value_vars=numerical_features, var_name='Variável', value_name='Valor')
plt.figure(figsize=(12, 6))
sns.boxplot(x='Variável', y='Valor', hue=categorical_feature, data=melted_df)
plt.title(f'Box Plot de Várias Variáveis Numéricas por {categorical_feature}')
plt.xticks(rotation=45)
st.pyplot(plt)
plt.clf()  # Limpa o gráfico para evitar sobreposição

# Box Plots Individuais para Cada Variável Numérica por Segmento
st.write("### Box Plots Individuais de Variáveis Numéricas por Segmento")
n = len(numerical_features)
fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 5 * n))
for i, var_numerica in enumerate(numerical_features):
    sns.boxplot(x=categorical_feature, y=var_numerica, data=dados_brutos, ax=axes[i])
    axes[i].set_title(f'Box Plot de {var_numerica} por {categorical_feature}')
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.clf()  # Limpa o gráfico após exibição

# Visualização de Clusters com Centroides
st.write("### Visualização de Clusters")
model_labels = {
    'DBSCAN1 Cluster': 'DBSCAN (eps=0.5, min_samples=5)',
    'DBSCAN2 Cluster': 'DBSCAN (eps=0.3, min_samples=10)',
    'KMeans1 Cluster': 'KMeans (3 clusters)',
    'KMeans2 Cluster': 'KMeans (4 clusters)',
    'Hierarchical1 Cluster': 'Agglomerative (3 clusters)',
    'Hierarchical2 Cluster': 'Agglomerative (4 clusters)'
}

# Plotar cada modelo de cluster
fig, axes = plt.subplots(len(model_labels), 2, figsize=(20, 30))
for i, (cluster_col, title) in enumerate(model_labels.items()):
    sns.scatterplot(data=dados_brutos, x='Volume transacional (R$)', y='Receita Transacional (R$)',
                    hue=cluster_col, palette='viridis', s=60, alpha=0.7, ax=axes[i, 0])
    axes[i, 0].set_title(f'{title} - Volume vs Receita')
    axes[i, 0].set_xlabel('Volume Transacional (R$)')
    axes[i, 0].set_ylabel('Receita Transacional (R$)')
    
    sns.scatterplot(data=dados_brutos, x='Receita Transacional (R$)', y='CAC (R$)_x',
                    hue=cluster_col, palette='viridis', s=60, alpha=0.7, ax=axes[i, 1])
    axes[i, 1].set_title(f'{title} - Receita vs CAC')
    axes[i, 1].set_xlabel('Receita Transacional (R$)')
    axes[i, 1].set_ylabel('CAC (R$)')
    
    # Adicionar centroides apenas para KMeans
    if "KMeans" in cluster_col:
        modelo_kmeans = modelos[cluster_col.lower().replace(" ", "").replace("cluster", "")]
        centroides = scaler.inverse_transform(modelo_kmeans.cluster_centers_)
        axes[i, 1].scatter(centroides[:, 1], centroides[:, 2], s=200, c='red', marker='X', label='Centroides')
        axes[i, 1].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
st.pyplot(fig)
