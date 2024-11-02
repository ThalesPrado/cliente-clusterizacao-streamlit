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

st.write("Por se tratar de técnica de aprendizado de máquina não supervisionado o modelo define os IDs de clientes que pertencem a cada grupo diferente do aprendizado de máquina supervisionado onde a classificação de cada cliente é previamente definida.")



# Etapa 1: Carregamento e Visualização dos Dados
st.header("Etapa 1: Carregamento e Visualização dos Dados")
st.write("Nesta etapa, foram carregados os dados brutos recebidos em formato de planila do excel com duas abas sendo a primeira chamada 'Base Transacional' e 'Base CAC'. Foram unificados na base transacional todos os dados relevantes para clusterização como as variáveis numéricas junto com seu respectivo CAC. Aqui podemos ver uma amostra para entender as variáveis disponíveis e o estado inicial dos dados brutos.")

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
    Durante a limpeza de dados o intuito foi identificar valores ausentes e realizar seu respectivos tratamentos,garantir que as colunas numéricas estivessem no formato adequado, identificar possíveis inconsistências nos segmentos e seleção de variáveis relevantes para a clusterização . Para uma base de dados com muitas variáveis pode se aplicar métodos de redução de dimensionalidade a fim de diminuir a complexidade do modelo, nessa solução essa etapa não foi realizada.
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
    Nesta etapa, foi realizado a análise descritiva para verificar as distribuições das variáveis numéricas e categóricas para entender melhor os dados. Aqui podemos visualizar as distribuições usando histogramas e identificar outliers. O que mais chamou a atenção nessa etapa foi a assimetria a esqueda dos dados que em termos práticos mostra que maiores frequências se encontram nos menores valores e grande quantidade de outliers independente do segmento ao qual aquela observação pertencesse.
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
st.subheader("Box Plot consolidado para todas as variáveis por segmento")
dados_melted = pd.melt(dados_limpos, id_vars=['Segmento'], value_vars=numeric_columns, 
                       var_name='Variável', value_name='Valor')
plt.figure(figsize=(14, 8))
sns.boxplot(x='Variável', y='Valor', hue='Segmento', data=dados_melted)
plt.title('Box Plot de Variáveis Numéricas por Segmento')
plt.xticks(rotation=45)
st.pyplot(plt)
plt.clf()

# Box Plot por Segmento
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Segmentos únicos na base de dados
segmentos_unicos = dados_limpos['Segmento'].unique()

# Para cada variável numérica, exibe um box plot separado por segmento
st.header("Box Plot de Variáveis Numéricas por Segmento")
for col in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dados_limpos, x='Segmento', y=col)
    plt.title(f'Distribuição de {col} por Segmento')
    plt.xlabel('Segmento')
    plt.ylabel(col)
    st.pyplot(plt)
    plt.clf()

# Base Sumarizada por ID e Segmento com Explicação
st.header("Etapa 4: Visualização da Base Sumarizada")
with st.expander("O que é a base sumarizada?"):
    st.write("""
    A base sumarizada é uma visão agregada dos dados originais, agrupados por ID e Segmento. Para conferir um rótulo de cluster a cada ID foram feitos agregações,logo os clientes que compraram nos meses 1 e 2 foram unificados de acordo com seus respectivos IDs em cada segmento. Dessa forma, a variável "mês" foi desconsiderada no processo de modelagem. Base conta com correlação positiva na maior parte das colunas com casos onde esses valores são significativos como o volume transacional e receita Transacional com valor de 89% que implica em alta correlação.
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
    st.header("Etapa 5: Modelagem e Métricas de Avaliação")
    st.write(""" Metodologia Kmeans: J = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2.
- De maneira geral o algoritmo tenta minimizar a soma das distâncias quadradas entre os pontos no plano e o centro do grupo ao qual pertencem. Onde Ci representa o conjunto de pontos que no nosso caso são  todos os clientes do i-ésimo cluster e nosso μi seria sua centroide dos pontos nesse cluster.
- Prós: Funciona bem em base de dados grandes e fácil de interpretar seus resultados.
- Contras: Requer espeficar o número de clusters K previamente e pode ter um desempenho baixo em clusters de formas não esféricas.
            """)
            
    st.write("### Bibliografia")
    st.write("""
1. Oliveira, L. S., & Sabourin, R. (2009). *Aprendizado de Máquina: Uma Abordagem Prática*
   
2. Ester, M., & Moro, M. M. (2013). *Análise de Agrupamentos para Mineração de Dados*.
   
3. Gama, J. (2015). *Tópicos em Mineração de Dados e Aprendizado de Máquina*. São Paulo: Pioneira Thomson Learning.
   
4. Rezende, S. O. (2005). *Sistemas Inteligentes: Fundamentos e Aplicações*. São Paulo: Manole.
   - Rezende explora o *k-means* em sistemas inteligentes, ressaltando a eficiência do algoritmo, mas também suas limitações com clusters não esféricos e a importância da escolha de \( K \).
""")

    st.write(""" Metodologia clusterização hierárquica: 
- De maneira geral o algoritmo tenta 
Esse método cria uma hierarquia de clusters através de fusão (aglomeração) ou divisão (divisão) sucessiva dos pontos. A distância entre clusters pode ser calculada de diferentes formas, como:

- Prós: Funciona bem em base de dados grandes e fácil de interpretar seus resultados.
- Contras: Requer espeficar o número de clusters K previamente e pode ter um desempenho baixo em clusters de formas não esféricas.
            """)
          
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

st.header("Etapa 6: Avaliação dos Resultados")

st.write("""
Na etapa final, avaliamos os resultados de clusterização para entender como os diferentes segmentos de clientes se comportam em relação aos clusters formados. Essa avaliação é feita com base nos gráficos de distribuição e nas médias e medianas de métricas importantes, como o Volume Transacional e o CAC. Como o modelo que melhor performou foi o modelo Kmeans com 3 cluster ele foi utilizado para apresentação dos gráficos abaixo.

### Análise dos Gráficos por Cluster e Segmento
1. **Total de Clientes por Cluster e Segmento**: Esse gráfico mostra a distribuição dos clientes em cada cluster, agrupados por segmento. A análise do total de clientes ajuda a identificar quais clusters contêm a maior parte dos clientes de cada segmento, o que pode indicar onde há concentração de clientes e onde a estratégia de retenção pode ser mais necessária.
   
2. **Média e Mediana de Volume Transacional e CAC por Cluster e Segmento**:
    - A **média** e a **mediana** de Volume Transacional permitem observar o valor médio e central das transações em cada cluster por segmento, indicando onde se concentram as transações mais altas e como isso varia entre segmentos. Clusters com volume transacional elevado sugerem segmentos mais lucrativos.

    - A **média** e a **mediana** de CAC mostram o custo médio de aquisição de clientes em cada cluster e segmento. Clusters com menor CAC indicam uma aquisição de clientes mais eficiente e potencialmente mais rentável.
   
3. **Distribuição de Volume Transacional e CAC por Cluster e Segmento (Box Plots)**:
    - Os **box plots** da distribuição do Volume Transacional e do CAC  por cluster e segmento ajudam a visualizar a variação dos valores dentro de cada cluster. Essa distribuição identifica a presença de outliers e a dispersão dos valores, fornecendo insights sobre a homogeneidade ou diversidade do comportamento dos clientes dentro de cada cluster.

    - Valores elevados e dispersos podem sugerir clientes de alta diversidade de comportamento, enquanto valores mais centralizados indicam segmentos e clusters com padrões mais estáveis.

4. **Distribuição de Receita Transacional por Cluster e Segmento (Box Plot)**:
    - A análise da **Receita Transacional ** por cluster e segmento permite identificar quais clusters contribuem mais para a receita e como essa contribuição varia entre os segmentos. Clusters com receitas altas são essenciais para o planejamento estratégico da empresa, indicando grupos de clientes de maior valor e que podem ser prioritários para estratégias de retenção e engajamento.
""")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Usando o modelo KMeans1 Cluster como exemplo
cluster_column = 'KMeans1 Cluster'

st.header("Análise dos Clusters por Segmento")

# Total de Clientes por Cluster e Segmento
st.subheader("Total de Clientes por Cluster e Segmento")
fig, ax = plt.subplots(figsize=(12, 8))
cluster_counts = df.groupby(['Segmento', cluster_column]).size().unstack().fillna(0)
cluster_counts.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Total de Clientes por Cluster e Segmento')
ax.set_xlabel('Segmento')
ax.set_ylabel('Número de Clientes')
ax.legend(title='Cluster')
st.pyplot(fig)

# Média e Mediana de Volume e CAC por Cluster e Segmento
mean_metrics = df.groupby(['Segmento', cluster_column])[['Volume transacional (R$)', 'CAC (R$)']].mean().unstack()
median_metrics = df.groupby(['Segmento', cluster_column])[['Volume transacional (R$)', 'CAC (R$)']].median().unstack()

# Plotar médias
st.subheader("Média de Volume Transacional e CAC por Cluster e Segmento")

fig, ax = plt.subplots(figsize=(12, 8))
mean_metrics['Volume transacional (R$)'].plot(kind='bar', ax=ax)
ax.set_title('Média de Volume Transacional por Cluster e Segmento')
ax.set_xlabel('Segmento')
ax.set_ylabel('Volume Transacional Médio (R$)')
ax.legend(title='Cluster')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(12, 8))
mean_metrics['CAC (R$)'].plot(kind='bar', ax=ax)
ax.set_title('Média de CAC por Cluster e Segmento')
ax.set_xlabel('Segmento')
ax.set_ylabel('CAC Médio (R$)')
ax.legend(title='Cluster')
st.pyplot(fig)

# Plotar medianas
st.subheader("Mediana de Volume Transacional e CAC por Cluster e Segmento")

fig, ax = plt.subplots(figsize=(12, 8))
median_metrics['Volume transacional (R$)'].plot(kind='bar', ax=ax)
ax.set_title('Mediana de Volume Transacional por Cluster e Segmento')
ax.set_xlabel('Segmento')
ax.set_ylabel('Volume Transacional Mediano (R$)')
ax.legend(title='Cluster')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(12, 8))
median_metrics['CAC (R$)'].plot(kind='bar', ax=ax)
ax.set_title('Mediana de CAC por Cluster e Segmento')
ax.set_xlabel('Segmento')
ax.set_ylabel('CAC Mediano (R$)')
ax.legend(title='Cluster')
st.pyplot(fig)

# Box Plot da Distribuição de Volume e CAC por Cluster e Segmento
st.subheader("Distribuição de Volume Transacional e CAC por Cluster e Segmento")

fig, ax = plt.subplots(figsize=(15, 8))
sns.boxplot(data=df, x=cluster_column, y='Volume transacional (R$)', hue='Segmento', ax=ax)
ax.set_title('Distribuição de Volume Transacional por Cluster e Segmento')
ax.set_xlabel('Cluster')
ax.set_ylabel('Volume Transacional (R$)')
ax.legend(title='Segmento', bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(15, 8))
sns.boxplot(data=df, x=cluster_column, y='CAC (R$)', hue='Segmento', ax=ax)
ax.set_title('Distribuição de CAC por Cluster e Segmento')
ax.set_xlabel('Cluster')
ax.set_ylabel('CAC (R$)')
ax.legend(title='Segmento', bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)

# Box Plot da Receita Transacional por Cluster e Segmento
st.subheader("Distribuição de Receita Transacional por Cluster e Segmento")

fig, ax = plt.subplots(figsize=(15, 8))
sns.boxplot(data=df, x=cluster_column, y='Receita Transacional (R$)', hue='Segmento', ax=ax)
ax.set_title('Distribuição de Receita Transacional por Cluster e Segmento')
ax.set_xlabel('Cluster')
ax.set_ylabel('Receita Transacional (R$)')
ax.legend(title='Segmento', bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)

import streamlit as st
import pandas as pd

# Suponha que `df` seja o DataFrame com os clientes e as colunas de cluster atribuídas pelo modelo
# Adicionamos a coluna de cluster a partir do modelo KMeans1

# Adicionando a coluna 'KMeans1 Cluster' à base sumarizada
# (Substitua 'df['KMeans1 Cluster']' pelo código necessário para gerar a coluna de cluster, se ainda não estiver na base)

st.header("Tabela de Clientes com Filtro por Segmento e Cluster")

# Filtros para Segmento e Cluster
segmento_options = df['Segmento'].unique()
cluster_options = df['KMeans1 Cluster'].unique()

selected_segmento = st.multiselect("Selecione o Segmento", options=segmento_options, default=segmento_options)
selected_cluster = st.multiselect("Selecione o Cluster", options=cluster_options, default=cluster_options)

# Aplicando filtros à tabela
filtered_df = df[(df['Segmento'].isin(selected_segmento)) & (df['KMeans1 Cluster'].isin(selected_cluster))]

# Exibindo a tabela filtrada com informações principais
st.write("### Clientes Classificados por Cluster e Segmento")
st.dataframe(filtered_df[['ID', 'Segmento', 'Volume transacional (R$)', 'Receita Transacional (R$)', 'CAC (R$)', 'KMeans1 Cluster']])

# Botão de download para os dados filtrados
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df_to_csv(filtered_df)

st.download_button(
    label="Baixar dados filtrados",
    data=csv_data,
    file_name='clientes_filtrados.csv',
    mime='text/csv',
)

st.header("Etapa 7: Recomendação")

st.write("""1) Qual seria sua sugestão de clusterização da base? Por quê?"

Melhor opção baseado nos modelos treinados foi o Kmeans com 3 clusters, uma vez que apresentou os seguintes scores.

Silhoutte score: 0.82. Esse score oscila entre 1 e -1 sendo que quanto maior o score melhor os pontos foram agrupados dentro do seu cluster.

Calinski score: 10212. Esse score informa que quanto maior mais os dados estão homogêneos dentro de um cluster indicando proximidade entre os IDs desse cluster e no caso do inter cluster mede a distância entre os centros dos clusters implicando que quanto maior distância mais distintos os clientes são uns dos outros,

Boldin score: 0.59. Esse score informa que quanto menor  mais separados os clientes de comportamentos diferentes estão.

""")

st.write("""2) Estando em um cenário de concorrência acirrada: qual estratégia comercial poderíamos implementar para diminuir nossa perda de clientes e quais clientes deveriamos priorizar para blindá-los de um eventual ataque da concorrência e Por quê?"

Melhor opção seria priorizar o segmento de restaurantes pois sozinho representam quase 40%* da base de clientes.Dentro da modelagem realizada quase 98%* dos clientes pertencem ao cluster 0 o que infelizmente implica em um volume transacional baixo em contrapartida apresentam baixo CAC.

Recomendação: Implementar políticas de cross-selling e upselling no segmento de restaurantes para aumentar o ticket médio do cluster 0.

Recomendação: Aplicação de política de descontos para fomentar o aumento do volume de vendas em períodos de baixa demanda no segmento de restaurantes do cluster 0.

Recomendação: Zerar ofertas para clientes do cluster com maior ticket médio como no caso dos cluster 1 e 2.
""")

st.write("""3) Dado o direcionamento de maximizar o lucro da Companhia, atual e futuro quais podem ser as sugestões?"

Recomendação: Reduzir CAC do cluster 0 via utilização de automação para interações de suporte e campanhas de marketing reduzindo o custo operacional atrelado a esse tipo de cluster de baixo retorno financeiro.

Recomendação: Otimizar preços dado a definição dos clusters pela modelagem visando aumento de margem 
com preços distintos para clientes com comportamento de compra distintos.
""")

st.write("""4) Quanto estima-se que teremos de receita nos meses seguintes se seguirmos as recomendações? 
""")
import streamlit as st
import pandas as pd

# Suponha que `filtered_df` seja a base de dados já filtrada e processada
# Filtro para Segmento com `key` única para evitar duplicação de ID
segmento_options = filtered_df['Segmento'].unique()
selected_segmento = st.multiselect("Selecione o Segmento", options=segmento_options, default=segmento_options, key="segmento_multiselect")

# Aplicando filtro de segmento
filtered_df_segmento = filtered_df[filtered_df['Segmento'].isin(selected_segmento)]

# Exibindo a tabela filtrada com informações principais
st.header("Base de Dados Inicial com Clientes Classificados por Cluster e Segmento")
st.dataframe(filtered_df_segmento[['ID', 'Segmento', 'Volume transacional (R$)', 'Receita Transacional (R$)', 'CAC (R$)', 'KMeans1 Cluster']])

# Agrupar dados para calcular o número de clientes e receita total por cluster (0, 1, 2)
cluster_summary = filtered_df_segmento.groupby('KMeans1 Cluster').agg({
    'ID': 'count',
    'Receita Transacional (R$)': 'sum'
}).rename(columns={'ID': 'Num_Clientes', 'Receita Transacional (R$)': 'Receita_Total'})

# Calcular ticket médio para cada cluster
cluster_summary['Ticket_Medio'] = cluster_summary['Receita_Total'] / cluster_summary['Num_Clientes']

# Exibir resumo inicial
st.write("### Resumo Inicial de Clientes e Receita por Cluster (0, 1, 2)")
st.write(cluster_summary[['Num_Clientes', 'Receita_Total', 'Ticket_Medio']])

# Guardar a receita total inicial
receita_total_inicial = cluster_summary['Receita_Total'].sum()

# Entradas para Migração entre Clusters a partir do Cluster 0
st.header("Simulação de Migração do Cluster 0 para os Clusters 1 e 2")
migra_0_para_1 = st.slider("Percentual de clientes migrando do Cluster 0 para o Cluster 1", min_value=0, max_value=100, value=10, key="migracao_0_para_1")
migra_0_para_2 = st.slider("Percentual de clientes migrando do Cluster 0 para o Cluster 2", min_value=0, max_value=100, value=5, key="migracao_0_para_2")

# Simulação de migração: aplicando mudanças nos números de clientes entre clusters
# Ajustando apenas o número de clientes
if 0 in cluster_summary.index and 1 in cluster_summary.index:
    # Calcular clientes migrando de Cluster 0 para 1 e arredondar para o inteiro mais próximo
    clientes_migrando_0_para_1 = round(cluster_summary.loc[0, 'Num_Clientes'] * (migra_0_para_1 / 100))
    cluster_summary.loc[0, 'Num_Clientes'] -= clientes_migrando_0_para_1
    cluster_summary.loc[1, 'Num_Clientes'] += clientes_migrando_0_para_1

if 0 in cluster_summary.index and 2 in cluster_summary.index:
    # Calcular clientes migrando de Cluster 0 para 2 e arredondar para o inteiro mais próximo
    clientes_migrando_0_para_2 = round(cluster_summary.loc[0, 'Num_Clientes'] * (migra_0_para_2 / 100))
    cluster_summary.loc[0, 'Num_Clientes'] -= clientes_migrando_0_para_2
    cluster_summary.loc[2, 'Num_Clientes'] += clientes_migrando_0_para_2

# Recalcular Receita Projetada após Migração
cluster_summary['Receita_Projetada'] = cluster_summary['Num_Clientes'] * cluster_summary['Ticket_Medio']

# Exibir tabela final com as receitas projetadas
st.write("### Projeção de Clientes e Receita por Cluster após Migração")
st.dataframe(cluster_summary[['Num_Clientes', 'Receita_Total', 'Ticket_Medio', 'Receita_Projetada']])

# Receita Total Projetada
receita_total_projetada = cluster_summary['Receita_Projetada'].sum()

# Comparação entre receita inicial e projetada
st.write(f"**Receita Total Inicial (Antes da Migração)**: R$ {receita_total_inicial:,.2f}")
st.write(f"**Receita Total Projetada (Após a Migração)**: R$ {receita_total_projetada:,.2f}")