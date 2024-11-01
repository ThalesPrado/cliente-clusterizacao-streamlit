import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração do estilo de visualização
sns.set(style="whitegrid")

# Título do Projeto
st.title("Clusterização de Clientes")
st.write("Este projeto realiza uma análise exploratória e uma clusterização de clientes para entender melhor o comportamento de diferentes segmentos.")

# Etapa 1: Carregamento e Visualização dos Dados
st.header("Etapa 1: Carregamento e Visualização dos Dados")
st.write("Nesta etapa, carregamos os dados brutos e exibimos uma amostra para entender as variáveis disponíveis e o estado inicial dos dados.")

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
    Durante a limpeza de dados, identificamos e tratamos valores ausentes, duplicados e garantimos que as colunas numéricas estão no formato correto.
    Também preenchemos valores nulos usando a média, caso necessário.
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
    Nesta etapa, exploramos as distribuições das variáveis numéricas e categóricas para entender melhor os dados. Visualizamos as distribuições usando histogramas e identificamos outliers com box plots.
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
    A base sumarizada é uma visão agregada dos dados originais, agrupados por ID e Segmento. Isso nos ajuda a visualizar dados agregados por cliente e segmento.
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


