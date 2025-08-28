# 2_fragmentacao_e_salvamento.py (VERSÃO COM FRAGMENTAÇÃO INTELIGENTE)

import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

pasta_textos = "textos_processados"
arquivo_saida_pkl = "fragmentos.pkl"

print("Iniciando a fragmentação inteligente dos textos...")

documentos_fragmentados = []

for nome_arquivo_txt in os.listdir(pasta_textos):
    if nome_arquivo_txt.endswith(".txt"):
        caminho_txt = os.path.join(pasta_textos, nome_arquivo_txt)

        with open(caminho_txt, "r", encoding="utf-8") as f:
            texto_completo = f.read()

        # --- INÍCIO DA MUDANÇA ---
        # Ajustamos o splitter para ser mais inteligente:
        # chunk_size=1000: Fragmentos menores e mais focados.
        # chunk_overlap=150: Uma boa sobreposição para não perder o contexto.
        # separators: Prioriza a quebra em parágrafos (\n\n), depois em linhas (\n),
        # depois em sentenças (". "), o que cria fragmentos muito mais coerentes.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""] # <<< Regra de separação mais inteligente
        )
        # --- FIM DA MUDANÇA ---

        fragmentos_texto = text_splitter.split_text(texto_completo)

        for frag_texto in fragmentos_texto:
            doc = Document(
                page_content=frag_texto,
                metadata={"source": nome_arquivo_txt}
            )
            documentos_fragmentados.append(doc)

        print(f"Arquivo '{nome_arquivo_txt}' foi dividido em {len(fragmentos_texto)} fragmentos mais inteligentes.")

with open(arquivo_saida_pkl, "wb") as f:
    pickle.dump(documentos_fragmentados, f)

print(f"\nFragmentos salvos com sucesso no arquivo: '{arquivo_saida_pkl}'")