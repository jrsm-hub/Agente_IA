import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
try:
    from langchain_chroma import Chroma
except (ImportError, ModuleNotFoundError):
    from langchain_community.vectorstores import Chroma
try:
    from langchain_core.tools import Tool
except (ImportError, ModuleNotFoundError):
    try:
        from langchain.tools import Tool
    except (ImportError, ModuleNotFoundError):
        from langchain.agents import Tool
from config import GOOGLE_API_KEY, EMBEDDING_MODEL, BANCO_VETORIAL_DIR

@st.cache_resource
def carregar_embeddings(api_key: str = None):
    """Inicializa o modelo de embeddings."""
    chave = (api_key or GOOGLE_API_KEY or "").strip()
    if not chave:
        st.error("Chave de API do Google (GOOGLE_API_KEY) não encontrada!")
        st.stop()
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=chave)

@st.cache_resource
def carregar_vectordb(api_key: str = None):
    """Inicializa e carrega o banco vetorial Chroma."""
    embeddings = carregar_embeddings(api_key=api_key)
    if not os.path.exists(BANCO_VETORIAL_DIR):
        print(f"Aviso: Diretório do banco vetorial '{BANCO_VETORIAL_DIR}' não encontrado.")
    return Chroma(persist_directory=BANCO_VETORIAL_DIR, embedding_function=embeddings)

def criar_rag_tool(vectordb, llm):
    """Cria a ferramenta RAG para consulta aos manuais de pesquisa com citação de fontes."""
    
    def executar_consulta_com_fontes(query: str) -> str:
        if isinstance(query, dict):
            query = query.get("query") or query.get("input") or str(query)
            
        docs = vectordb.similarity_search(str(query), k=3)
        if not docs:
            return "Nenhuma informação relevante encontrada nos manuais disponíveis."
            
        contexto = "\n\n".join([
            f"[Fonte: {doc.metadata.get('source', 'Manual')}]\n{doc.page_content}"
            for doc in docs
        ])
        
        prompt = (
            f"Você é um consultor acadêmico. Com base exclusivamente nos trechos de manuais fornecidos abaixo, "
            f"responda à consulta de forma fundamentada e objetiva em português do Brasil.\n\n"
            f"TRECHOS DOS MANUAIS:\n{contexto}\n\n"
            f"CONSULTA: {query}\n\n"
            f"RESPOSTA:"
        )
        
        from services.ai_service import extrair_texto_conteudo, assegurar_modelo_valido
        llm_valido = assegurar_modelo_valido(llm)
        try:
            resposta_raw = llm_valido.invoke(prompt)
        except Exception as e:
            if "404" in str(e) or "NOT_FOUND" in str(e) or "2.5" in str(e):
                llm_valido.model = "gemini-3.5-flash-lite"
                resposta_raw = llm_valido.invoke(prompt)
            else:
                raise e
        resposta = extrair_texto_conteudo(resposta_raw)
        
        fontes = sorted(list(set(doc.metadata.get('source', 'Manual') for doc in docs if doc.metadata.get('source'))))
        if fontes:
            fontes_str = ", ".join([f"`{f}`" for f in fontes])
            resposta += f"\n\n📚 **Manuais consultados:** {fontes_str}"
            
        return resposta

    return Tool(
        name="Consulta_Manuais_Pesquisa",
        func=executar_consulta_com_fontes,
        description="Use esta ferramenta para consultar manuais sobre metodologia científica, normas acadêmicas, estrutura de trabalhos e boas práticas de pesquisa."
    )
