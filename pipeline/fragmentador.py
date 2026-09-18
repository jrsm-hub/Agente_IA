import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import TEXTOS_PROCESSADOS_DIR, FRAGMENTOS_PKL_PATH

def fragmentar_textos(
    pasta_textos=TEXTOS_PROCESSADOS_DIR,
    arquivo_saida=FRAGMENTOS_PKL_PATH,
    chunk_size=1000,
    chunk_overlap=150
):
    """Divide os textos processados em fragmentos (chunks) otimizados para RAG."""
    if not os.path.exists(pasta_textos):
        print(f"Pasta '{pasta_textos}' não encontrada.")
        return []

    print("Iniciando a fragmentação inteligente dos textos...")
    documentos_fragmentados = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for nome_arquivo_txt in os.listdir(pasta_textos):
        if nome_arquivo_txt.endswith(".txt"):
            caminho_txt = os.path.join(pasta_textos, nome_arquivo_txt)

            with open(caminho_txt, "r", encoding="utf-8") as f:
                texto_completo = f.read()

            fragmentos_texto = text_splitter.split_text(texto_completo)

            for frag_texto in fragmentos_texto:
                doc = Document(
                    page_content=frag_texto,
                    metadata={"source": nome_arquivo_txt}
                )
                documentos_fragmentados.append(doc)

            print(f"Arquivo '{nome_arquivo_txt}' dividido em {len(fragmentos_texto)} fragmentos.")

    with open(arquivo_saida, "wb") as f:
        pickle.dump(documentos_fragmentados, f)

    print(f"Total de {len(documentos_fragmentados)} fragmentos salvos em '{arquivo_saida}'")
    return documentos_fragmentados

if __name__ == "__main__":
    fragmentar_textos()
