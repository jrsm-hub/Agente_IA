# 3_vetorizacao_e_db.py (VERSÃO ATUALIZADA)

import os
import pickle # Biblioteca para carregar objetos Python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma

# --- INÍCIO DA CONFIGURAÇÃO ---
GOOGLE_API_KEY = "AIzaSyCKvrA9lEiiQCaZGhEmSJ2r0pZ9yCo2hDY"
arquivo_entrada_pkl = "fragmentos.pkl"
pasta_db_vetorial = "banco_vetorial_chroma"
# --- FIM DA CONFIGURAÇÃO ---


# --- ETAPA 1: Carregar os Documentos Fragmentados do Arquivo ---
print(f"Carregando fragmentos do arquivo '{arquivo_entrada_pkl}'...")
with open(arquivo_entrada_pkl, "rb") as f:
    documentos_carregados = pickle.load(f)

print(f"{len(documentos_carregados)} fragmentos foram carregados.")

# Separa o conteúdo dos metadados para o ChromaDB
textos_dos_fragmentos = [doc.page_content for doc in documentos_carregados]
metadados_dos_fragmentos = [doc.metadata for doc in documentos_carregados]


# --- ETAPA 2: Inicializar o Modelo de Embeddings ---
print("Inicializando o modelo de embeddings do Google...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)


# --- ETAPA 3: Criar e Persistir o Banco de Dados Vetorial ---
print("Criando o banco de dados vetorial com ChromaDB...")
vectordb = Chroma.from_texts(
    texts=textos_dos_fragmentos,
    embedding=embeddings,
    metadatas=metadados_dos_fragmentos,
    persist_directory=pasta_db_vetorial
)
print(f"Banco de dados vetorial criado e salvo em '{pasta_db_vetorial}'")


# --- ETAPA 4: Testar a Busca (Exemplo) ---
print("\n--- Testando a busca por similaridade ---")
query = "Como evitar o plágio acadêmico?"
docs_encontrados = vectordb.similarity_search(query, k=2)

print(f"Pergunta do usuário: '{query}'")
print("\nFragmentos mais relevantes encontrados:")
for i, doc in enumerate(docs_encontrados):
    print(f"\n--- Resultado {i+1} ---")
    print(f"Fonte: {doc.metadata.get('source', 'N/A')}")
    print(f"Conteúdo: {doc.page_content[:300]}...")