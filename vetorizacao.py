

import os
import pickle
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- INÍCIO DA CONFIGURAÇÃO SEGURA ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Chave GOOGLE_API_KEY não encontrada. Verifique o seu ficheiro .env")

arquivo_entrada_pkl = "fragmentos.pkl"
pasta_db_vetorial = "banco_vetorial_chroma"

# Aumentamos o tamanho do lote para enviar mais dados de uma vez, tornando o processo mais rápido.
tamanho_lote = 100 


# ETAPA 1: Carregar os Documentos Fragmentados
print(f"A carregar fragmentos do ficheiro '{arquivo_entrada_pkl}'...")
with open(arquivo_entrada_pkl, "rb") as f:
    documentos_carregados = pickle.load(f)
print(f"{len(documentos_carregados)} fragmentos foram carregados.")

# ETAPA 2: Inicializar o Modelo de Embeddings
print("A inicializar o modelo de embeddings do Google...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# ETAPA 3: Criar o Banco de Dados Vetorial em Lotes
print(f"A criar o banco de dados vetorial em lotes de {tamanho_lote} fragmentos...")

# Apaga a base de dados antiga, se existir, para garantir que começamos do zero
if os.path.exists(pasta_db_vetorial):
    import shutil
    print(f"A apagar a base de dados antiga em '{pasta_db_vetorial}'...")
    shutil.rmtree(pasta_db_vetorial)

# Inicializa o banco de dados com o primeiro lote
primeiro_lote = documentos_carregados[:tamanho_lote]
vectordb = Chroma.from_documents(
    documents=primeiro_lote,
    embedding=embeddings,
    persist_directory=pasta_db_vetorial
)

# Processa os lotes restantes
for i in range(tamanho_lote, len(documentos_carregados), tamanho_lote):
    lote_atual = documentos_carregados[i : i + tamanho_lote]
    print(f"A processar lote {i//tamanho_lote + 1} de {len(documentos_carregados)//tamanho_lote + 1}...")
    
    vectordb.add_documents(lote_atual)
   
    
print(f"\nBanco de dados vetorial criado e salvo com sucesso em '{pasta_db_vetorial}'")

# ETAPA 4: Testar a Busca (Exemplo)
print("\n--- A testar a busca por similaridade ---")
query = "Como evitar o plágio académico?"
docs_encontrados = vectordb.similarity_search(query, k=2)

print(f"Pergunta do utilizador: '{query}'")
print("\nFragmentos mais relevantes encontrados:")
for i, doc in enumerate(docs_encontrados):
    print(f"\n--- Resultado {i+1} ---")
    print(f"Fonte: {doc.metadata.get('source', 'N/A')}")
    print(f"Conteúdo: {doc.page_content[:300]}...")