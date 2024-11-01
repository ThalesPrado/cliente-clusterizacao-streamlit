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
numeric_columns = ['Volume transacional (R$)', 'Receita Transacional (R$)', 'Receita Antecipação de recebíveis (R$)']
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
    'Receita Antecipação de recebíveis (R$)': 'sum'
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

# Configurar estilo e título do app
st.title("Análise de Clusterização de Clientes")
st.write("""
    Esta aplicação realiza diferentes modelos de clusterização para segmentar clientes. A escolha de modelos variados ajuda a entender qual deles melhor representa a estrutura dos dados.
    """)
st.write("### Etapas Realizadas")
st.write("""
1. Normalização dos dados: uso de `RobustScaler` para lidar com outliers.
2. Aplicação de três métodos de clusterização (KMeans, DBSCAN, e Aglomerativo).
3. Avaliação dos modelos utilizando métricas de avaliação: Índice de Silhouette, Índice de Calinski-Harabasz e Índice de Davies-Bouldin.
""")


st.header("Etapa 4: Visualização da Base Sumarizada")
with st.expander("O que é a base sumarizada?"):
    st.write("""
    A base sumarizada é uma visão agregada dos dados originais, agrupados por ID e Segmento. Para conferir um rótulo de cluster a cada ID, independentemente do mês, os clientes que compraram nos meses 1 e 2 foram unificados de acordo com seus respectivos IDs em cada segmento. Dessa forma, a variável "mês" foi desconsiderada no processo de modelagem. Base conta com correlação positiva na maior parte das colunas com casos com valores altos como o volume transacional e receita Transacional com valor de 89% que implica em alta correlação.
    """)

# Carregar dados e normalizar
@st.cache_data
def carregar_dados():
    return pd.read_excel("/mnt/data/Base_Missão_S.xlsx", engine="openpyxl")

df = carregar_dados()
features = df[['Volume transacional (R$)', 'Receita Transacional (R$)', 'CAC (R$)_x']]
scaler = RobustScaler()
features_scaled = scaler.fit_transform(features)

# Função para calcular métricas de avaliação
def evaluate_clustering(labels, data):
    silhouette = silhouette_score(data, labels) if len(set(labels)) > 1 else None
    calinski_harabasz = calinski_harabasz_score(data, labels) if len(set(labels)) > 1 else None
    davies_bouldin = davies_bouldin_score(data, labels) if len(set(labels)) > 1 else None
    return silhouette, calinski_harabasz, davies_bouldin

# Aplicar e avaliar modelos
results = {}

# Explicando a escolha de cada modelo e métricas
st.header("Modelos de Clusterização e Métricas de Avaliação")
st.write("""
   **Modelos Utilizados**:
   - **KMeans**: Algoritmo clássico de clusterização. Define clusters esféricos e é eficiente em termos computacionais, ideal para casos em que sabemos a quantidade de clusters esperados.
   - **DBSCAN**: Algoritmo baseado em densidade, ótimo para detectar clusters de formas arbitrárias e identificar outliers. Funciona bem para dados com densidade variável, mas é sensível aos parâmetros.
   - **Clusterização Aglomerativa**: Modelo hierárquico que não exige especificação inicial de número de clusters e pode revelar hierarquias naturais nos dados. No entanto, pode ser menos eficiente para grandes volumes de dados.
   
   **Métricas de Avaliação**:
   - **Índice de Silhouette**: Avalia a separação entre clusters; valores maiores indicam melhor definição dos clusters.
   - **Índice de Calinski-Harabasz**: Mede a relação entre dispersão intracluster e intercluster. Quanto maior, melhor a separação entre clusters.
   - **Índice de Davies-Bouldin**: Indica a similaridade média entre cada cluster e o mais semelhante; valores menores indicam melhor qualidade de clusters.
   """)

# Rodando Modelos
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
st.write("O método do cotovelo ajuda a identificar o número ideal de clusters para o KMeans, ao verificar onde a inércia começa a se estabilizar.")
wcss = []
k_values = range(1, 5)
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