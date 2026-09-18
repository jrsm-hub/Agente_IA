import os
import pickle
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import GOOGLE_API_KEY, EMBEDDING_MODEL, FRAGMENTOS_PKL_PATH, BANCO_VETORIAL_DIR

def vetorizar_documentos(
    arquivo_fragmentos=FRAGMENTOS_PKL_PATH,
    pasta_db=BANCO_VETORIAL_DIR,
    tamanho_lote=100
):
    """Gera embeddings e salva os fragmentos no banco vetorial Chroma."""
    if not GOOGLE_API_KEY:
        raise ValueError("Chave GOOGLE_API_KEY não encontrada no ambiente (.env).")

    if not os.path.exists(arquivo_fragmentos):
        raise FileNotFoundError(f"Arquivo de fragmentos '{arquivo_fragmentos}' não encontrado.")

    print(f"Carregando fragmentos de '{arquivo_fragmentos}'...")
    with open(arquivo_fragmentos, "rb") as f:
        documentos_carregados = pickle.load(f)
    print(f"{len(documentos_carregados)} fragmentos carregados.")

    print("Inicializando modelo de embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)

    # Remove o banco antigo se existir para reconstrução limpa
    if os.path.exists(pasta_db):
        print(f"Removendo banco vetorial antigo em '{pasta_db}'...")
        shutil.rmtree(pasta_db)

    print(f"Criando banco vetorial em lotes de {tamanho_lote}...")
    primeiro_lote = documentos_carregados[:tamanho_lote]
    vectordb = Chroma.from_documents(
        documents=primeiro_lote,
        embedding=embeddings,
        persist_directory=pasta_db
    )

    total_lotes = (len(documentos_carregados) + tamanho_lote - 1) // tamanho_lote
    for i in range(tamanho_lote, len(documentos_carregados), tamanho_lote):
        lote_atual = documentos_carregados[i : i + tamanho_lote]
        lote_num = (i // tamanho_lote) + 1
        print(f"Processando lote {lote_num} de {total_lotes}...")
        vectordb.add_documents(lote_atual)

    print(f"Banco vetorial criado com sucesso em '{pasta_db}'!")
    return vectordb

if __name__ == "__main__":
    vetorizar_documentos()
