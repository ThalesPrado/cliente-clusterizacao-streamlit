import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

# Carregar os modelos de clusterização
modelos = {
    "KMeans3": joblib.load("modelos/kmeans3.joblib"),
    "KMeans4": joblib.load("modelos/kmeans4.joblib"),
    "DBSCAN1": joblib.load("modelos/dbscan1.joblib"),
    "DBSCAN2": joblib.load("modelos/dbscan2.joblib"),
    "Hierarchical3": joblib.load("modelos/hierarchical3.joblib"),
    "Hierarchical4": joblib.load("modelos/hierarchical4.joblib")
}
scaler = RobustScaler()  # Ajuste para o scaler usado durante o treinamento

# Carregar a base de dados bruta
@st.cache
def carregar_dados():
    return pd.read_excel("base_dados_bruta.xlsx")  # Substitua pelo nome correto da sua base de dados bruta

# Função para sumarizar a base de dados
@st.cache
def sumarizar_dados(dados_brutos):
    # Agrupar por 'ID' e 'Segmento', somando as colunas relevantes
    dados_sumarizados = dados_brutos.groupby(['ID', 'Segmento']).agg({
        'Volume transacional (R$)': 'sum',
        'Receita Transacional (R$)': 'sum',
        'CAC (R$)': 'sum'
    }).reset_index()
    return dados_sumarizados

# Carregar e sumarizar a base de dados
dados_brutos = carregar_dados()
dados_sumarizados = sumarizar_dados(dados_brutos)

# Interface do Streamlit
st.title("Classificação de Clientes por Cluster")
st.write("Escolha um modelo e insira os dados sumarizados do cliente para obter o cluster correspondente.")

# Filtro por Segmento
segmentos_disponiveis = dados_sumarizados['Segmento'].unique()
segmento_selecionado = st.selectbox("Escolha o segmento do cliente", segmentos_disponiveis)

# Filtrar a base de dados sumarizada pelo segmento selecionado
dados_segmento = dados_sumarizados[dados_sumarizados['Segmento'] == segmento_selecionado]

# Escolha do modelo de clusterização
modelo_selecionado = st.selectbox("Escolha o modelo de clusterização", list(modelos.keys()))

# Input dos dados sumarizados do cliente
volume = st.number_input("Volume Transacional (R$)", min_value=0.0)
receita = st.number_input("Receita Transacional (R$)", min_value=0.0)
cac = st.number_input("CAC (R$)", min_value=0.0)

# Exibir a base de dados sumarizada filtrada pelo segmento selecionado (opcional)
if st.checkbox("Mostrar base de dados sumarizada pelo segmento"):
    st.write(dados_segmento.head())

# Classificar Cliente ao clicar no botão
if st.button("Classificar Cliente"):
    # Pré-processar os dados
    dados_cliente = pd.DataFrame([[volume, receita, cac]], columns=['Volume transacional (R$)', 'Receita Transacional (R$)', 'CAC (R$)'])
    dados_escalados = scaler.transform(dados_cliente)

    # Obter o modelo e fazer a previsão
    modelo = modelos[modelo_selecionado]
    if modelo_selecionado.startswith("DBSCAN"):
        cluster = modelo.fit_predict(dados_escalados)
    else:
        cluster = modelo.predict(dados_escalados)
    
    # Mostrar o cluster ao usuário
    st.success(f"O cliente do segmento '{segmento_selecionado}' foi classificado no Cluster: {int(cluster[0])}")
