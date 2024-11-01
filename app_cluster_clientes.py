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

# Identificar valores ausentes
st.write("#### Valores Ausentes")
st.write(dados_brutos.isnull().sum())

# Remover duplicatas, se houver
st.write("#### Remoção de Duplicatas")
dados_limpos = dados_brutos.drop_duplicates()
st.write(f"Número de linhas após remoção de duplicatas: {dados_limpos.shape[0]}")

# Visualizar tipos de dados
st.write("#### Tipos de Dados e Conversão")
st.write(dados_limpos.dtypes)

# Converter colunas numéricas, se necessário (exemplo: garantir que valores estão no tipo correto)
# Ajuste os nomes das colunas conforme necessário
numeric_columns = ['Volume transacional (R$)', 'Receita Transacional (R$)', 'Receita Antecipação de recebíveis (R$)', 'CAC (R$)_x']
for col in numeric_columns:
    dados_limpos[col] = pd.to_numeric(dados_limpos[col], errors='coerce')

# Reavaliar valores ausentes após conversão
st.write("#### Valores Ausentes Após Conversão")
st.write(dados_limpos.isnull().sum())

# Preencher ou remover valores ausentes, conforme necessário
# Exemplo: preenchendo valores ausentes com a média das colunas numéricas
for col in numeric_columns:
    if dados_limpos[col].isnull().sum() > 0:
        dados_limpos[col].fillna(dados_limpos[col].mean(), inplace=True)

# EDA - Visualização de Distribuições
st.write("### Análise Exploratória de Dados (EDA)")

# Visualização das distribuições das variáveis numéricas
st.write("#### Distribuição das Variáveis Numéricas")
for col in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(dados_limpos[col], kde=True)
    plt.title(f'Distribuição de {col}')
    st.pyplot(plt)
    plt.clf()  # Limpa o gráfico após exibição

# Visualização das variáveis categóricas
st.write("#### Distribuição das Variáveis Categóricas")
categorical_columns = ['Segmento']  # Ajuste conforme necessário
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=dados_limpos[col])
    plt.title(f'Distribuição de {col}')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.clf()  # Limpa o gráfico após exibição

# Exibir a base de dados limpa para referência
st.write("### Dados Limpos")
st.write(dados_limpos.head())
