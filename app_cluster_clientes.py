import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração do estilo de visualização
sns.set(style="whitegrid")

# Título do Projeto
st.title("Clusterização de Clientes")
st.write("Este projeto visa realizar o processo de clusterização de clientes para cada um dos diferentes segmentos e baseado nisso desenvolver estratégias afim de maximizar a rentabilidade da empresa.O intuito do projeto é responder ou pelo menos ter um direcionamento para os seguintes problemas")
st.write("1) Qual seria sua sugestão de clusterização da base? Por quê?")
st.write("2) Estando em um cenário de concorrência acirrada: qual estratégia comercial poderíamos implementar para diminuir nossa perda de clientes e quais clientes deveriamos priorizar para blindá-los de um eventual ataque da concorrência e Por quê?")
st.write("3) Dado o direcionamento de maximizar o lucro da Companhia, atual e futuro quais podem ser as sugestões?")
st.write("4) Quanto estima-se que teremos de receita nos meses seguintes se seguirmos as recomendações? da equipe de dados")

# Etapa 1: Carregamento e Visualização dos Dados
st.header("Etapa 1: Carregamento e Visualização dos Dados")
st.write("Nesta etapa, foram carregados os dados brutos recebidos em formato de planila do excel com duas abas sendo a primeira chamada 'Base Transacional' e 'Base CAC'.Foram unificados na base transacional todos os dados relevantes para clusterização como as variáveis numéricas junto com seu respectivo CAC. Aqui podemos ver uma amostra para entender as variáveis disponíveis e o estado inicial dos dados.")

@st.cache_data
def carregar_dados():
    # Substitua o caminho pelo nome correto do arquivo Excel
    return pd.read_excel("Base_Missão_S.xlsx", engine="openpyxl")

dados_brutos = carregar_dados()
st.write("### Amostra da Base de Dados")
st.write(dados_brutos.head())

# Limpeza de Dados com Explicação
st.header("Etapa 2: Limpeza de Dados")
with st.expander("Explicação"):
    st.write("""
    Durante a limpeza de dados,o intuito foi identificar valores ausentes e realizar seu respectivo tratamento,garantir que as colunas numéricas estivessem no formato adequado, identificar possíveis inconsistências nos segmentos entre outros.
    """)

dados_limpos = dados_brutos.drop_duplicates().copy()
numeric_columns = ['Volume transacional (R$)', 'Receita Transacional (R$)', 'Receita Antecipação de recebíveis (R$)','CAC (R$)']
if 'CAC (R$)_x' in dados_limpos.columns:
    numeric_columns.append('CAC (R$)_x')

for col in numeric_columns:
    dados_limpos[col] = pd.to_numeric(dados_limpos[col], errors='coerce')
    dados_limpos[col].fillna(dados_limpos[col].mean(), inplace=True)

# Análise Exploratória de Dados (EDA)
st.header("Etapa 3: Análise Exploratória de Dados (EDA)")
st.write("""
    Nesta etapa, foi realizado a análise descritiva para verificar as distribuições das variáveis numéricas e categóricas para entender melhor os dados. Visualizamos as distribuições usando histogramas e identificamos outliers com box plots, matrizes entre outros. O mais interessante nessa etapa foi verificar a quantidade de outliers que existem na base, assimetria dos dados de cada variável e etc.
""")

# Distribuições das Variáveis Numéricas
st.subheader("Distribuição das Variáveis Numéricas")
for col in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(dados_limpos[col], kde=True)
    plt.title(f'Distribuição de {col}')
    st.pyplot(plt)
    plt.clf()

# Box Plot Consolidado
st.subheader("Box Plot Consolidado para Todas as Variáveis por Segmento")
dados_melted = pd.melt(dados_limpos, id_vars=['Segmento'], value_vars=numeric_columns, 
                       var_name='Variável', value_name='Valor')
plt.figure(figsize=(14, 8))
sns.boxplot(x='Variável', y='Valor', hue='Segmento', data=dados_melted)
plt.title('Box Plot de Variáveis Numéricas por Segmento')
plt.xticks(rotation=45)
st.pyplot(plt)
plt.clf()

# Base Sumarizada por ID e Segmento com Explicação
st.header("Etapa 4: Visualização da Base Sumarizada")
with st.expander("O que é a base sumarizada?"):
    st.write("""
    A base sumarizada é uma visão agregada dos dados originais, agrupados por ID e Segmento. Para conferir um rótulo de cluster a cada ID, independentemente do mês, os clientes que compraram nos meses 1 e 2 foram unificados de acordo com seus respectivos IDs em cada segmento. Dessa forma, a variável "mês" foi desconsiderada no processo de modelagem. Base conta com correlação positiva na maior parte das colunas com casos com valores altos como o volume transacional e receita Transacional com valor de 89% que implica em alta correlação.
    """)

dados_sumarizados = dados_limpos.groupby(['ID', 'Segmento']).agg({
    'Volume transacional (R$)': 'sum',
    'Receita Transacional (R$)': 'sum',
    'Receita Antecipação de recebíveis (R$)': 'sum', 'CAC (R$)': 'sum'
})
if 'CAC (R$)_x' in dados_limpos.columns:
    dados_sumarizados['CAC (R$)_x'] = dados_limpos.groupby(['ID', 'Segmento'])['CAC (R$)_x'].sum()

dados_sumarizados.reset_index(inplace=True)
st.write(dados_sumarizados.head())

# Importando o numpy para corrigir o erro de NameError
import numpy as np

# Removendo a coluna 'Segmento' para garantir que apenas dados numéricos sejam considerados na matriz de correlação
dados_sumarizados_numericos = dados_sumarizados.select_dtypes(include=[np.number])

# Calculando a matriz de correlação apenas com as colunas numéricas
correlation_matrix = dados_sumarizados_numericos.corr()

# Plotando a matriz de correlação ajustada
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, square=True)
plt.title("Matriz de Correlação dos Dados Sumarizados")
st.pyplot(plt)  # Exibe a matriz de correlação no Streamlit
plt.clf()  # Limpa a figura para evitar sobreposição

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração do estilo de visualização
sns.set(style="whitegrid")

# Verificar se a base sumarizada já existe
if 'dados_sumarizados' in globals():
    df = dados_sumarizados  # Usar a base sumarizada existente
    st.write("### Amostra da Base Sumarizada para realizar a modelagem")
    st.write(df.head())
    
    # Seleção dinâmica de colunas numéricas, incluindo 'CAC (R$)' apenas se existir
    numeric_columns = ['Volume transacional (R$)', 'Receita Transacional (R$)', 'Receita Antecipação de recebíveis (R$)']
    if 'CAC (R$)' in df.columns:
        numeric_columns.append('CAC (R$)')
    
    # Extração das colunas numéricas para a clusterização
    features = df[numeric_columns]
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)

    # Função para calcular métricas de avaliação
    def evaluate_clustering(labels, data):
        silhouette = silhouette_score(data, labels) if len(set(labels)) > 1 else None
        calinski_harabasz = calinski_harabasz_score(data, labels) if len(set(labels)) > 1 else None
        davies_bouldin = davies_bouldin_score(data, labels) if len(set(labels)) > 1 else None
        return silhouette, calinski_harabasz, davies_bouldin

    results = {}

    # Explicação das escolhas de modelos e métricas
    st.header("Etapa 5: Modelagem de Métricas de Avaliação")
    st.write("""
       **Modelos Utilizados**:
       - **KMeans**: Efetivo para clusters esféricos e de tamanho similar. Ideal quando o número de clusters é conhecido.
       - **DBSCAN**: Identifica clusters de forma arbitrária e detecta outliers, útil para dados com densidade variável.
       - **Clusterização Aglomerativa**: Modelo hierárquico, não exige número inicial de clusters e revela hierarquias naturais, mas pode ser lento para grandes dados.
       
       **Métricas de Avaliação**:
       - **Índice de Silhouette**: Avalia a separação entre clusters; valores maiores indicam melhor definição.
       - **Índice de Calinski-Harabasz**: Mede dispersão intra e interclusters. Quanto maior, melhor a separação.
       - **Índice de Davies-Bouldin**: Mede a similaridade entre clusters; valores menores indicam clusters melhor definidos.
       """)

    # Aplicação dos Modelos de Clusterização
    # DBSCAN
    dbscan1 = DBSCAN(eps=0.5, min_samples=5)
    dbscan1_labels = dbscan1.fit_predict(features_scaled)
    results['DBSCAN1'] = evaluate_clustering(dbscan1_labels, features_scaled)
    df['DBSCAN1 Cluster'] = dbscan1_labels

    dbscan2 = DBSCAN(eps=0.3, min_samples=10)
    dbscan2_labels = dbscan2.fit_predict(features_scaled)
    results['DBSCAN2'] = evaluate_clustering(dbscan2_labels, features_scaled)
    df['DBSCAN2 Cluster'] = dbscan2_labels

    # KMeans
    kmeans1 = KMeans(n_clusters=3, random_state=0)
    kmeans1_labels = kmeans1.fit_predict(features_scaled)
    results['KMeans1'] = evaluate_clustering(kmeans1_labels, features_scaled)
    df['KMeans1 Cluster'] = kmeans1_labels

    kmeans2 = KMeans(n_clusters=4, random_state=0)
    kmeans2_labels = kmeans2.fit_predict(features_scaled)
    results['KMeans2'] = evaluate_clustering(kmeans2_labels, features_scaled)
    df['KMeans2 Cluster'] = kmeans2_labels

    # Aglomerativo
    hierarchical1 = AgglomerativeClustering(n_clusters=3)
    hierarchical1_labels = hierarchical1.fit_predict(features_scaled)
    results['Hierarchical1'] = evaluate_clustering(hierarchical1_labels, features_scaled)
    df['Hierarchical1 Cluster'] = hierarchical1_labels

    hierarchical2 = AgglomerativeClustering(n_clusters=4)
    hierarchical2_labels = hierarchical2.fit_predict(features_scaled)
    results['Hierarchical2'] = evaluate_clustering(hierarchical2_labels, features_scaled)
    df['Hierarchical2 Cluster'] = hierarchical2_labels

    # Resultados das métricas de cada modelo
    metrics_df = pd.DataFrame(results, index=['Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index']).T
    st.write("### Comparação das Métricas de Avaliação dos Modelos")
    st.write(metrics_df)

    # Análise do Método do Cotovelo para KMeans
    st.write("### Análise do Método do Cotovelo para o KMeans")
    wcss = []
    k_values = range(1, 6)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(features_scaled)
        wcss.append(kmeans.inertia_)

    # Plot do Método do Cotovelo
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss, marker='o')
    plt.title('Método do Cotovelo para Determinar o Número de Clusters em KMeans')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Soma dos Erros Quadrados (WCSS)')
    st.pyplot(plt)
    plt.clf()
else:
    st.write("Base sumarizada não encontrada. Por favor, verifique o carregamento dos dados.")

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Configurar o tamanho da figura para visualizações no Streamlit
st.header("Visualização dos Resultados de Clusterização")

# Dicionário para labels e títulos de clusters
model_labels = {
    'DBSCAN1 Cluster': 'DBSCAN (eps=0.5, min_samples=5)',
    'DBSCAN2 Cluster': 'DBSCAN (eps=0.3, min_samples=10)',
    'KMeans1 Cluster': 'KMeans (3 clusters)',
    'KMeans2 Cluster': 'KMeans (4 clusters)',
    'Hierarchical1 Cluster': 'Agglomerative (3 clusters)',
    'Hierarchical2 Cluster': 'Agglomerative (4 clusters)'
}

# Plotar cada modelo
for cluster_col, title in model_labels.items():
    # Plot 1: Volume transacional vs Receita Transacional
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df, x='Volume transacional (R$)', y='Receita Transacional (R$)',
                    hue=cluster_col, palette='viridis', s=60, alpha=0.7, ax=ax)
    ax.set_title(f'{title} - Volume vs Receita')
    ax.set_xlabel('Volume Transacional (R$)')
    ax.set_ylabel('Receita Transacional (R$)')
    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)  # Exibir no Streamlit
    
    # Plot 2: Receita Transacional vs CAC
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df, x='Receita Transacional (R$)', y='CAC (R$)',
                    hue=cluster_col, palette='viridis', s=60, alpha=0.7, ax=ax)
    ax.set_title(f'{title} - Receita vs CAC')
    ax.set_xlabel('Receita Transacional (R$)')
    ax.set_ylabel('CAC (R$)')
    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)  # Exibir no Streamlit