# app.py (VERSÃO FINAL COM AGENTE GEMINI-TOOLS)

import os
import streamlit as st
from dotenv import load_dotenv

# Importações para o novo agente Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# Importações para o cache e ferramentas
import langchain
from langchain.cache import InMemoryCache
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

import nest_asyncio

# --- CONFIGURAÇÃO E CARREGAMENTO DE RECURSOS ---
nest_asyncio.apply()
load_dotenv()
langchain.llm_cache = InMemoryCache()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Chave de API do Google não encontrada!")
    st.stop()

@st.cache_resource
def carregar_recursos():
    print("Carregando recursos...")
    # Usamos um modelo que suporta tool calling
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=0.7)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vectordb = Chroma(persist_directory="banco_vetorial_chroma", embedding_function=embeddings)
    print("Recursos carregados.")
    return llm, vectordb

llm, vectordb = carregar_recursos()

# --- DEFINIÇÃO DAS FERRAMENTAS (COM SINTAXE MODERNA) ---
# Criamos a ferramenta RAG de uma forma mais robusta
retriever = vectordb.as_retriever()
rag_tool = create_retriever_tool(
    retriever,
    "Consulta_Manuais_TCC",
    "Use esta ferramenta para responder perguntas específicas sobre metodologia científica, escrita de TCC, plágio, etc."
)

tools = [rag_tool]

# --- CONSTRUÇÃO DO AGENTE GEMINI-TOOLS ---
# Este prompt é muito mais simples e direto
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente de TCC amigável e prestativo chamado 'Orientador Virtual'. Sua missão é ajudar estudantes com suas pesquisas acadêmicas. Responda sempre em português do Brasil."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Criamos o agente otimizado para tool calling
agent = create_tool_calling_agent(llm, tools, prompt)

# O AgentExecutor une o agente com as ferramentas
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# --- INTERFACE DO STREAMLIT ---
st.set_page_config(page_title="Orientador Virtual Avançado", page_icon="🎓")
st.title("🎓 Orientador Virtual Avançado")
st.info("Agora uso as capacidades nativas do Gemini para te dar a melhor resposta.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_usuario := st.chat_input("Qual a sua dúvida?"):
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Raciocinando com o Gemini..."):
            try:
                # O histórico agora é passado dentro do invoke
                response = agent_executor.invoke({
                    "input": prompt_usuario
                })
                resposta = response["output"]
                st.markdown(resposta)
            except Exception as e:
                resposta = f"Desculpe, ocorreu um erro: {e}"
                st.error(resposta)
    
    st.session_state.messages.append({"role": "assistant", "content": resposta})