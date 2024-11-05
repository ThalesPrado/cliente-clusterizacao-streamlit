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
st.write("2) Estando em um cenário de concorrência acirrada: qual estratégia comercial poderíamos implementar para diminuir nossa perda de clientes e quais clientes deveriamos priorizar para blindá-los de um eventual ataque da concorrência e por quê?")
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
    st.write(""" **Metodologia Clusterização por Distância**:

- De maneira geral o algoritmo tenta minimizar a soma das distâncias quadradas entre os pontos no plano e o centro do grupo ao qual pertencem. Onde Ci representa o conjunto de pontos que no nosso caso são  todos os clientes do i-ésimo cluster e nosso μi seria sua centroide dos pontos deste cluster.

- Algoritmo baseado em distância euclidiana, onde estabelece primeiramente as centroides e posteriormente agrupa esses clientes/pontos em torno desses centros. 

- Prós: Funciona bem em base de dados grandes e é fácil de interpretar seus resultados.

- Contras: Requer espeficar o número de clusters K previamente e pode ter um desempenho baixo em clusters de formas não esféricas.
            """)

    # Exibe a fórmula matemática com `st.latex`
    st.latex(r"J = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2")

    st.write("Onde J é a função de custo a minimizar.")
    st.write("Somatório externo percorre cada cluster Ci indo de i = 1 até i = K, sendo K o total de clusters que se deseja formar")
    st.write("Somatório da distância de todos os pontos x pertencentes ao cluster Ci dos centros/centroides μi.")
    st.write("Calculo da distância ao quadrado entre um ponto x do cluster Ci e centroide de μi.")

    st.write("### Bibliografia")
    st.write("""
1. Oliveira, L. S., & Sabourin, R. (2009). *Aprendizado de Máquina: Uma Abordagem Prática*
   
2. Ester, M., & Moro, M. M. (2013). *Análise de Agrupamentos para Mineração de Dados*.
   
3. Gama, J. (2015). *Tópicos em Mineração de Dados e Aprendizado de Máquina*.
   
4. Rezende, S. O. (2005). *Sistemas Inteligentes: Fundamentos e Aplicações*.
""")

    
    st.write(""" **Metodologia Clusterização Hierárquica**: 
- De maneira geral o algoritmo trabalha com a menor distância entre qualquer par de pontos em dois clusters Ci e Cj ou a maior distância entre qualquer par de pontos em Ci e Cj ou distância media entre todos os pontos de Ci e Cj e termina o processo realizando a fusão onde verifica os pares com os menores valores entre si e faz a composição deles em um cluster maior. 

- O processo sempre inicia com o total n de cluster, no nosso caso como temos 9000 clientes o algoritmo inicia com 9000 clusters. Esse processo de cálculo de distâncias, escolha dos clusters mais próximos e fusão continua até que seja atingido o número desejado de clusters K, alternativamente o processo acaba quando todos os pontos estão unidos.

- O algoritmo pode ser aglomerativo, onde inicia com n pontos e depois efetua os agrupamentos ou divisivo que inicia com n = 1 como se fosse um grande cluster e a cada iteração se divide visando chegar em um número de clusters ideal.

- Prós: Não requer o número de clusters antecipadamente e funciona bem em pequenos conjuntos de dados.

- Contras: Sensível a ruídos e outliers.
            """)

    import streamlit as st

    st.write("#### Distância Mínima (Single Linkage)")
    st.latex(r"d_{min}(C_i, C_j) = \min_{x \in C_i, y \in C_j} \|x - y\|")

    st.write("Onde o dmin(Ci,Cj) representa a distância mínima entre dois cluster")
    st.write("Onde a norma ∥x−y∥ indica a distância entre dois pontos x e y.")

    st.write("Intuição: Imaginemos que contamos com dois grupos de clientes. A distância mínima considera os clientes mais próximos entre esses dois grupos para medir a distância entre os clusters. Em cada etapa o algoritmo procurar unir os clusters que têm a menor distância mínima entre si.")

    st.write("#### Distância Máxima (Complete Linkage)")
    st.latex(r"d_{max}(C_i, C_j) = \max_{x \in C_i, y \in C_j} \|x - y\|")

    st.write("Onde o dmax(Ci,Cj) representa a distância máxima entre dois cluster")
    st.write("Onde a norma ∥x−y∥ indica a distância entre dois pontos x e y.")

    st.write("Intuição: Esta abordagem considera o par de pontos mais distante entre os clusters para medir sua distância.A distância máxima é a distância entre os dois clientes mais distantes um de cada cluster/grupo, esse valor que vai definir a separação.")

    st.write("#### Distância Média (Average Linkage)")
    st.latex(r"d_{avg}(C_i, C_j) = \frac{1}{|C_i| \cdot |C_j|} \sum_{x \in C_i} \sum_{y \in C_j} \|x - y\|")

    st.write("Onde o davg(Ci,Cj) representa o calculo da distância média entre todos os pares de pontos entre dois clusters.")
    st.write("Onde a norma ∥x−y∥ indica a distância entre dois pontos x e y.")

    st.write("Intuição: Esta abordagem calcula a distância média entre os dois grupos. Isso leva em conta todos os pontos nos clusters e fornece essa média da distância entre eles.")

    st.write("### Bibliografia")
    st.write("""
1. Han, J., Kamber, M., & Pei, J. (2012). *Data Mining: Conceitos e Técnicas*.

2. Alpaydin, E. (2016). *Aprendizado de Máquina*.

3. Tan, P.-N., Steinbach, M., & Kumar, V. (2009). *Introdução à Mineração de Dados*.

4. Domingos, P. (2015). *Machine Learning: A Arte e a Ciência de Algoritmos que Aprendem*.

5. Morais, H. (2018). *Estatística e Análise de Dados com R*.

6. Hair Jr., J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2009). *Análise Multivariada de Dados*.
""")

    st.write(""" **Metodologia Clusterização por Densidade**: 
- De maneira geral o algoritmo DBSCAN como outras vertentes que se baseam em densidade definem clusters como regiões de alta densidade de pontos separadas por áreas de baixa densidade. Conta com dois parâmetros principais sendo o ε a distância máxima para considerar um ponto dentro da vizinhança de outro e MinPts sendo o número mínimo de pontos necessários para que uma região seja considerada um cluster.

- O algoritmo identifica agrupamentos como áreas densamente populosas ignorando áreas com poucos pontos. 

- Prós: Não requer o número de clusters antecipadamente e trabalha bem com dados outliers.

- Contras: Sensível à escolha de Eps e MinPts.
            """)

# Vizinhança de ε
    st.write("#### 1. Vizinhança de ε")
    st.latex(r"N_{\epsilon}(p) = \{ q \in D \mid \| p - q \| \leq \epsilon \}")
    st.write("Todos os pontos q no conjunto de dados D que estão a uma distância de no máximo ε de p formam a vizinhança de p.")

    st.write("Onde Nε(p) representa a vizinhança ε do ponto p, essa é a região ao redor do ponto central p que inclui todos os outros pontos em um cluster respeitando uma distância ε.")

    st.write("Intuição: Em termos práticos vamos imaginar que colocamos um círculo de raio ϵ em volta de cada cliente,todos os outros clientes que caem nessa região são próximos. Se temos um ϵ = 30 cm, isso indica que para dois clientes estarem no mesmo cluster eles precisam estar a uma distância de no máximo de 30 cm.")

# Ponto Central
    st.write("#### 2. Ponto Central")
    st.latex(r"|N_{\epsilon}(p)| \geq \text{MinPts}")
    st.write("Um ponto p é considerado um ponto central se o número de pontos na sua vizinhança de ε for maior ou igual a MinPts.")
    st.write("Intuição: Número mínimo de clientes que precisam estar perto uns dos outros para que possamos formar um cluster. Supondo um MinPts = 5, isso significaria que devemos ter pelo menos 5 clientes próximos uns dos outros para ser um grupo/cluster.")

# Ponto de Borda
    st.write("#### 3. Ponto de Borda")
    st.latex(r"q \in N_{\epsilon}(p) \quad \text{e} \quad |N_{\epsilon}(q)| < \text{MinPts}")
    st.write("Um ponto q é um ponto de borda se ele está na vizinhança de um ponto central p, mas não tem MinPts vizinhos em sua própria vizinhança.")
    st.write("Intuição: Ponto de borda trata dos clientes que estão perto do grupo/cluster principal mas não contam com clientes suficientes para serem pontos centrais.")

# Ponto de Ruído
    st.write("#### 4. Ponto de Ruído")
    st.latex(r"|N_{\epsilon}(r)| < \text{MinPts} \quad \text{e} \quad r \notin N_{\epsilon}(p) \text{ para todo } p \text{ ponto central}")
    st.write("Um ponto r é ruído se ele não possui MinPts vizinhos e não está na vizinhança de nenhum ponto central, sendo considerado um outlier.")
    st.write("Intuição: Casos de clientes completamente isolados estando longe do cluster principal e de outros clientes, logo não pode ser incluído em nenhum grupo.")

    st.write("### Bibliografia")
    st.write("""
1. Han, J., Kamber, M., & Pei, J. (2012). *Data Mining: Conceitos e Técnicas*.

2. Alpaydin, E. (2016). *Aprendizado de Máquina*.

3. Tan, P.-N., Steinbach, M., & Kumar, V. (2009). *Introdução à Mineração de Dados*.

4. Domingos, P. (2015). *Machine Learning: A Arte e a Ciência de Algoritmos que Aprendem*.

5. Morais, H. (2018). *Estatística e Análise de Dados com R*.

6. Hair Jr., J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2009). *Análise Multivariada de Dados*.
""")

    st.title("Métricas de Avaliação de Clusterização")

    st.write("### 1. Índice de Silhouette")
    st.latex(r"s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i)}")
    st.write("""
- **Intuição**: Mede o quão bem definido está cada ponto dentro de seu cluster. Valores próximos de 1 indicam que os pontos estão bem ajustados ao seu cluster e longe de outros clusters.
- **Valores**:
   - **Próximos de 1**: Boa coesão e separação.
   - **Próximos de 0**: Ponto na fronteira entre clusters.
   - **Negativos**: Pontos possivelmente no cluster errado.
""")

    st.write("### 2. Índice de Calinski-Harabasz")
    st.latex(r"CH = \frac{\text{Tr}(B_k)}{\text{Tr}(W_k)} \times \frac{N - k}{k - 1}")
    st.write("""
- **Intuição**: Avalia a relação entre a dispersão interna e a separação entre clusters. Quanto maior o valor, melhor a separação entre clusters.
- **Valores**:
   - **Valores Altos**: Clusters bem separados e coesos.
   - **Valores Baixos**: Clusters muito próximos ou dispersos.
""")

    st.write("### 3. Índice de Davies-Bouldin")
    st.latex(r"DBI = \frac{1}{k} \sum_{i=1}^k \max_{j \neq i} \frac{S_i + S_j}{M_{i,j}}")
    st.write("""
- **Intuição**: Mede a similaridade entre clusters. Valores menores indicam clusters mais bem definidos e separados.
- **Valores**:
   - **Valores Baixos**: Clusters compactos e bem separados.
   - **Valores Altos**: Clusters mal definidos ou sobrepostos.
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

# Adicione este código ao final da Etapa 5
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

# **Define KMeans com 3 clusters como padrão**
default_model = 'KMeans1 Cluster'

# Selecionar o modelo de clusterização com KMeans 3 clusters como padrão
selected_model = st.selectbox("Selecione o modelo de clusterização para visualizar os gráficos de dispersão:", 
                              options=list(model_labels.keys()), 
                              index=list(model_labels.keys()).index(default_model),  # Define KMeans1 Cluster como padrão
                              format_func=lambda x: model_labels[x])

# **Define "restaurante" como o segmento padrão**
segment_options = df['Segmento'].unique()
default_segment = 'restaurante' if 'restaurante' in segment_options else segment_options[0]

# Filtrar segmentos com "restaurante" como padrão
selected_segments = st.multiselect("Selecione o(s) segmento(s) para visualizar:", 
                                   options=segment_options, 
                                   default=[default_segment])

# Filtra os dados com base nos segmentos selecionados
filtered_data = df[df['Segmento'].isin(selected_segments)]

# Exibir os gráficos de dispersão para o modelo e segmentos selecionados
if selected_model and not filtered_data.empty:
    title = model_labels[selected_model]
    
    # Plot 1: Volume transacional vs Receita Transacional
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=filtered_data, x='Volume transacional (R$)', y='Receita Transacional (R$)',
                    hue=selected_model, palette='viridis', s=60, alpha=0.7, ax=ax)
    ax.set_title(f'{title} - Volume vs Receita (Segmentos selecionados)')
    ax.set_xlabel('Volume Transacional (R$)')
    ax.set_ylabel('Receita Transacional (R$)')
    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)  # Exibir no Streamlit
    
    # Plot 2: Receita Transacional vs CAC
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=filtered_data, x='Receita Transacional (R$)', y='CAC (R$)',
                    hue=selected_model, palette='viridis', s=60, alpha=0.7, ax=ax)
    ax.set_title(f'{title} - Receita vs CAC (Segmentos selecionados)')
    ax.set_xlabel('Receita Transacional (R$)')
    ax.set_ylabel('CAC (R$)')
    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)  # Exibir no Streamlit

    st.write("""
Na etapa final, avaliamos os resultados de clusterização para entender como os diferentes segmentos de clientes se comportam em relação aos clusters formados. Essa avaliação é feita com base nos gráficos de distribuição e nas médias e medianas de métricas importantes, como o Volume Transacional e o CAC. Como o modelo que melhor performou foi o modelo Kmeans com 3 clusters ele foi utilizado para apresentação dos gráficos abaixo.

### Análise dos Gráficos por Cluster e Segmento
1. **Total de Clientes por Cluster e Segmento**: 
    - Esse gráfico mostra a distribuição dos clientes em cada cluster, agrupados por segmento. A análise do total de clientes ajuda a identificar quais clusters contêm a maior parte dos clientes de cada segmento, o que pode indicar onde há concentração de clientes e onde a estratégia de retenção pode ser mais necessária.
   
2. **Média e Mediana de Volume Transacional e CAC por Cluster e Segmento**:
    - A média e a mediana de Volume Transacional permitem observar o valor médio e central das transações em cada cluster por segmento, indicando onde se concentram as transações mais altas e como isso varia entre segmentos. Clusters com volume transacional elevado sugerem segmentos mais lucrativos.

    - A média e a mediana de CAC mostram o custo médio de aquisição de clientes em cada cluster e segmento. Clusters com menor CAC indicam uma aquisição de clientes mais eficiente e potencialmente mais rentável.
   
3. **Distribuição de Volume Transacional e CAC por Cluster e Segmento (Box Plots)**:
    - Os box plots da distribuição do Volume Transacional e do CAC  por cluster e segmento ajudam a visualizar a variação dos valores dentro de cada cluster. Essa distribuição identifica a presença de outliers e a dispersão dos valores, fornecendo insights sobre a homogeneidade ou diversidade do comportamento dos clientes dentro de cada cluster.

    - Valores elevados e dispersos podem sugerir clientes de alta diversidade de comportamento, enquanto valores mais centralizados indicam segmentos e clusters com padrões mais estáveis.

4. **Distribuição de Receita Transacional por Cluster e Segmento (Box Plot)**:
    - A análise da Receita Transacional por cluster e segmento permite identificar quais clusters contribuem mais para a receita e como essa contribuição varia entre os segmentos. Clusters com receitas altas são essenciais para o planejamento estratégico da empresa, indicando grupos de clientes de maior valor e que podem ser prioritários para estratégias de retenção e engajamento.
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

# Box Plot da Distribuição de Volume Transacional e CAC por Cluster e Segmento
st.header("Distribuição de Volume Transacional e CAC por Cluster e Segmento")

# Plot para Distribuição de Volume Transacional por Cluster e Segmento
st.subheader("Distribuição de Volume Transacional por Cluster e Segmento")
fig, ax = plt.subplots(figsize=(15, 8))
sns.boxplot(data=df, x=cluster_column, y='Volume transacional (R$)', hue='Segmento', ax=ax)
ax.set_title('Distribuição de Volume Transacional por Cluster e Segmento')
ax.set_xlabel('Cluster')
ax.set_ylabel('Volume Transacional (R$)')
ax.legend(title='Segmento', bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)  # Exibir o gráfico de Volume Transacional

# Plot para Distribuição de CAC por Cluster e Segmento
st.subheader("Distribuição de CAC por Cluster e Segmento")
fig, ax = plt.subplots(figsize=(15, 8))
sns.boxplot(data=df, x=cluster_column, y='CAC (R$)', hue='Segmento', ax=ax)
ax.set_title('Distribuição de CAC por Cluster e Segmento')
ax.set_xlabel('Cluster')
ax.set_ylabel('CAC (R$)')
ax.legend(title='Segmento', bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)  # Exibir o gráfico de CAC

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

    -   Silhoutte score: 0.82. 
    
    Esse score oscila entre 1 e -1 sendo que quanto maior o score melhor os pontos foram agrupados dentro do seu cluster.

    -   Calinski score: 10212. 
    
    Esse score informa que quanto maior mais os dados estão homogêneos dentro de um cluster indicando proximidade entre os IDs desse cluster e no caso do inter cluster mede a distância entre os centros dos clusters implicando que quanto maior distância mais distintos os clientes são uns dos outros.

    -   Boldin score: 0.59. 
    
    Esse score informa que quanto menor  mais separados os clientes de comportamentos diferentes estão.

""")

st.write("""2) Estando em um cenário de concorrência acirrada: qual estratégia comercial poderíamos implementar para diminuir nossa perda de clientes e quais clientes deveriamos priorizar para blindá-los de um eventual ataque da concorrência e Por quê?

    -   Melhor opção seria priorizar o segmento de restaurantes pois sozinho representam quase 40%** da base de clientes.Dentro da modelagem realizada quase 98%* dos clientes pertencem ao cluster 0 o que infelizmente implica em um volume transacional baixo em contrapartida apresentam baixo CAC.

    -   1.Recomendação : Criar um programa de fidelidade ofertando descontos diferentes para canais diferentes como delivery,lojas entre outros canais dentro desse segmento. O intuito é manter o volume de vendas com comunicação personalizada por cada canal e ofertar preços diferentes dado a demanda como no caso do Happy Hour entre outras possibilidades.

    -   2.Recomendação: Trabalhar com pacotes para grupos corporativos, eventos e outras festividades especiais visando incentivar aumentar a base de clientes.

    -   3.Recomendação: Trabalhar em parceria com outros players de mercado. Clientes que compram em nossa lojas de Restaurantes ganham benefícios em outros segmentos do grupo como Posto de Gasolina, Supermercado e etc.

    -   4.Recomendação: Ter uma agenda de eventos para grupos específicos e pop-up com Chefs com estrela Michellin.
""")

st.write("""3) Dado o direcionamento de maximizar o lucro da Companhia, atual e futuro quais podem ser as sugestões?"

    -   1.Recomendação: Para reduzir o CAC pode-se trabalhar com retargeting de clientes que tiverem alguma integração com nosso app ou site.

    -   2.Recomendação: Realizar teste A/B para inferir canais que apresentam maior custo-benefício e focar os investimentos de marketing neles evitando desperdício de recursos em canais de baixo retorno.
    
    -   3.Recomendação: Implementar políticas de cross-selling e upselling para aumentar o ticket médio do cluster 0. Isso serve para campanhas de marketing e comunicação de modo geral ofertando melhor personalização para cada agrupamento.
""")

st.write("""4) Quanto estima-se que teremos de receita nos meses seguintes se seguirmos as recomendações? 
""")
import streamlit as st
import pandas as pd

# Suponha que `filtered_df` seja a base de dados já filtrada e processada
# Define as opções de segmentos únicos
segmento_options = filtered_df['Segmento'].unique()

# Define "Restaurante" como padrão, se ele existir na lista de opções
default_segmento = ["Restaurante"] if "Restaurante" in segmento_options else segmento_options

# Cria o multiselect com "Restaurante" como valor padrão, utilizando uma única `key`
selected_segmento = st.multiselect(
    "Selecione o Segmento",
    options=segmento_options,
    default=default_segmento,
    key="segmento_multiselect_unique"
)

# Aplicando filtro de segmento com base na seleção
filtered_df_segmento = filtered_df[filtered_df['Segmento'].isin(selected_segmento)]

# Exibindo a tabela filtrada com informações principais
st.write("### Base de Dados Filtrada por Segmento")
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