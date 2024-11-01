import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração do estilo de visualização
sns.set(style="whitegrid")

# Função para carregar a base de dados
@st.cache_data
def carregar_dados():
    # Substitua o caminho pelo nome correto do arquivo Excel
    return pd.read_excel("Base_Missão_S.xlsx", engine="openpyxl")

# Carregar a base de dados
dados_brutos = carregar_dados()

# Exibir uma amostra dos dados
st.write("### Amostra da Base de Dados")
st.write(dados_brutos.head())

# Limpeza de Dados
st.write("### Limpeza de Dados")
dados_limpos = dados_brutos.drop_duplicates().copy()

# Verificação das colunas presentes
st.write("#### Colunas Disponíveis")
st.write(dados_limpos.columns)

# Definir colunas numéricas com base nas colunas disponíveis
numeric_columns = ['Volume transacional (R$)', 'Receita Transacional (R$)', 'Receita Antecipação de recebíveis (R$)']
if 'CAC (R$)_x' in dados_limpos.columns:
    numeric_columns.append('CAC (R$)_x')

# Conversão de colunas numéricas e tratamento de valores ausentes
for col in numeric_columns:
    dados_limpos[col] = pd.to_numeric(dados_limpos[col], errors='coerce')
    dados_limpos[col].fillna(dados_limpos[col].mean(), inplace=True)

# EDA - Distribuições
st.write("#### Distribuição das Variáveis Numéricas")
for col in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(dados_limpos[col], kde=True)
    plt.title(f'Distribuição de {col}')
    st.pyplot(plt)
    plt.clf()

st.write("#### Distribuição das Variáveis Categóricas")
categorical_columns = ['Segmento']
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=dados_limpos[col])
    plt.title(f'Distribuição de {col}')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.clf()

# EDA - Box Plots por Segmento para Identificação de Outliers
st.write("### Box Plots por Segmento para Identificação de Outliers")
for col in numeric_columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Segmento', y=col, data=dados_limpos)
    plt.title(f'Box Plot de {col} por Segmento')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.clf()

# Base Sumarizada por ID e Segmento
st.write("### Base Sumarizada por ID e Segmento")
dados_sumarizados = dados_limpos.groupby(['ID', 'Segmento']).agg({
    'Volume transacional (R$)': 'sum',
    'Receita Transacional (R$)': 'sum',
    'Receita Antecipação de recebíveis (R$)': 'sum'
})

# Adicionar CAC se existir na base
if 'CAC (R$)_x' in dados_limpos.columns:
    dados_sumarizados['CAC (R$)_x'] = dados_limpos.groupby(['ID', 'Segmento'])['CAC (R$)_x'].sum()

dados_sumarizados.reset_index(inplace=True)
st.write(dados_sumarizados.head())


