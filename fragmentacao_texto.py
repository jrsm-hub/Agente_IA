# 2_fragmentacao_e_salvamento.py (VERSÃO ATUALIZADA)

import os
import pickle  # Biblioteca para salvar objetos Python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Nomes das pastas e arquivos
pasta_textos = "textos_processados"
arquivo_saida_pkl = "fragmentos.pkl" # .pkl é a extensão padrão para arquivos pickle

print("Iniciando a fragmentação dos textos...")

documentos_fragmentados = []

for nome_arquivo_txt in os.listdir(pasta_textos):
    if nome_arquivo_txt.endswith(".txt"):
        caminho_txt = os.path.join(pasta_textos, nome_arquivo_txt)

        with open(caminho_txt, "r", encoding="utf-8") as f:
            texto_completo = f.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )

        # Divide o texto em strings
        fragmentos_texto = text_splitter.split_text(texto_completo)

        # Converte as strings em objetos Document do LangChain, adicionando os metadados
        for frag_texto in fragmentos_texto:
            doc = Document(
                page_content=frag_texto,
                metadata={"source": nome_arquivo_txt}
            )
            documentos_fragmentados.append(doc)

        print(f"Arquivo '{nome_arquivo_txt}' foi dividido em {len(fragmentos_texto)} fragmentos.")

# --- INÍCIO DA PARTE NOVA: SALVANDO OS FRAGMENTOS ---
print(f"\nSalvando {len(documentos_fragmentados)} fragmentos no arquivo '{arquivo_saida_pkl}'...")

# "wb" significa que estamos escrevendo ('w') em modo binário ('b'), que é o que o pickle precisa
with open(arquivo_saida_pkl, "wb") as f:
    pickle.dump(documentos_fragmentados, f)

print("Fragmentos salvos com sucesso!")
# --- FIM DA PARTE NOVA ---