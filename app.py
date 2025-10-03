import os
import streamlit as st
from dotenv import load_dotenv

# Importações principais do LangChain para o agente de diálogo
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.memory import ConversationBufferMemory

# Importações para as ferramentas e a geração de documento
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import nest_asyncio

# --- CONFIGURAÇÃO E CARREGAMENTO DE RECURSOS ---
nest_asyncio.apply()
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@st.cache_resource
def carregar_recursos():
    if not GOOGLE_API_KEY:
        st.error("Chave de API do Google não encontrada. Verifique o seu ficheiro .env!")
        st.stop()
    print("A inicializar os recursos...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.8)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vectordb = Chroma(persist_directory="banco_vetorial_chroma", embedding_function=embeddings)
    print("Recursos inicializados.")
    return llm, vectordb

llm, vectordb = carregar_recursos()

# --- INICIALIZAÇÃO DO ESTADO DA APLICAÇÃO ---
if "fase" not in st.session_state:
    st.session_state.fase = "COLETA"
    st.session_state.historico_mensagens = [{
        "role": "assistant",
        "content": "Olá! Sou o seu estrategista de pesquisa. Vamos construir juntos a base para o seu trabalho académico. Para começar, qual é a sua grande área de interesse?"
    }]
    st.session_state.agent_executor = None
    st.session_state.documento_gerado = None

# --- LÓGICA DO AGENTE DINÂMICO (FASE 1) ---
def gerar_proxima_pergunta_dinamica(historico_texto):
    template = """
    Você é um orientador de pesquisa a conduzir uma entrevista. Analise o HISTÓRICO DA CONVERSA e formule a PRÓXIMA PERGUNTA para ajudar o aluno a afunilar as suas ideias (Tema -> Problema -> Objetivo -> Metodologia). A sua pergunta deve oferecer de 3 a 4 opções concretas.
    HISTÓRICO DA CONVERSA: {historico}
    PRÓXIMA PERGUNTA DO ORIENTADOR (com opções):
    """
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(historico=historico_texto)

# --- LÓGICA DA GERAÇÃO DO DOCUMENTO (FASE 2) ---
def gerar_documento_estrategico(historico_texto):
    template = """
    Você é um orientador de pesquisa experiente. Com base em toda a entrevista no HISTÓRICO DA CONVERSA, crie um documento estratégico para o aluno.

    HISTÓRICO DA CONVERSA: {historico}

    **ESTRUTURA OBRIGATÓRIA (use Markdown):**
    # Estratégia de Pesquisa para o seu Trabalho Académico
    ## 1. Análise do seu Perfil de Pesquisa
    ## 2. Caminhos de Pesquisa Sugeridos
    ### Caminho A: [Título]
    - Descrição, Por que é promissor, Primeiros Passos, Riscos.
    ### Caminho B: [Título]
    - Descrição, Por que é promissor, Primeiros Passos, Riscos.
    ## 3. Conclusão e Recomendações
    """
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)
    with st.spinner("O estrategista está a consolidar a nossa conversa e a construir o seu documento..."):
        documento_final = chain.run(historico=historico_texto)
        st.session_state.documento_gerado = documento_final

# --- LÓGICA DO AGENTE DE DIÁLOGO ABERTO (FASE 3) ---
def inicializar_agente_de_dialogo():
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever())
    tools = [
        Tool(name="Consulta_Manuais_Pesquisa", func=rag_chain.run, description="Use para perguntas específicas sobre metodologia, escrita, plágio, etc."),
    ]
    
    prompt = hub.pull("hwchase17/react-chat")
    prompt.template = prompt.template.replace(
        "You are a helpful assistant. Respond to the user's request as best you can.",
        "Você é um assistente de pesquisa a continuar uma conversa com um aluno. O histórico contém a entrevista e o documento estratégico já fornecido. Continue a ajudar o aluno. **RESPONDA SEMPRE EM PORTUGUÊS DO BRASIL.**"
    ).replace("Begin!", "Comece!").replace("Thought:", "Pensamento:").replace("Action:", "Ação:").replace("Action Input:", "Entrada da Ação:").replace("Observation:", "Observação:")

    agent = create_react_agent(llm, tools, prompt)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Popula a memória com o histórico completo
    for msg in st.session_state.historico_mensagens:
        if msg["role"] == "user":
            memory.chat_memory.add_user_message(msg["content"])
        else:
            memory.chat_memory.add_ai_message(msg["content"])
            
    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)

# --- INTERFACE PRINCIPAL DO STREAMLIT ---
st.set_page_config(page_title="Estrategista de Pesquisa Académica", page_icon="🧭")
st.title("🧭 Estrategista de Pesquisa Académica")

# Exibe o histórico da conversa
for msg in st.session_state.historico_mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Fase 1: Coleta
if st.session_state.fase == "COLETA":
    if len(st.session_state.historico_mensagens) > 1:
        if st.button("Já temos informação suficiente. Gerar Documento Estratégico!"):
            st.session_state.fase = "GERACAO"
            st.rerun()
    if prompt_usuario := st.chat_input("A sua resposta..."):
        st.session_state.historico_mensagens.append({"role": "user", "content": prompt_usuario})
        with st.chat_message("user"): st.markdown(prompt_usuario)
        historico_texto = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.historico_mensagens])
        with st.chat_message("assistant"):
            with st.spinner("A formular a próxima pergunta..."):
                proxima_pergunta = gerar_proxima_pergunta_dinamica(historico_texto)
                st.markdown(proxima_pergunta)
                st.session_state.historico_mensagens.append({"role": "assistant", "content": proxima_pergunta})
        st.rerun()

# Fase 2: Geração
elif st.session_state.fase == "GERACAO":
    if st.session_state.documento_gerado is None:
        historico_texto = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.historico_mensagens])
        gerar_documento_estrategico(historico_texto)
        # Adiciona o documento e a mensagem de transição ao histórico
        st.session_state.historico_mensagens.append({"role": "assistant", "content": st.session_state.documento_gerado})
        st.rerun() # Recarrega para mostrar o documento
    else:
        # Após mostrar o documento, exibe o botão para a próxima fase
        if st.button("Excelente! Agora vamos discutir esta estratégia"):
            st.session_state.fase = "DIALOGO_ABERTO"
            st.session_state.agent_executor = inicializar_agente_de_dialogo()
            # Adiciona a mensagem final da fase de estratégia
            msg_transicao = "A sua estratégia de pesquisa está acima. Agora, estou pronto para discutir e aprofundar os pontos. O que gostaria de explorar primeiro?"
            st.session_state.historico_mensagens.append({"role": "assistant", "content": msg_transicao})
            st.rerun()

# Fase 3: Diálogo Aberto
elif st.session_state.fase == "DIALOGO_ABERTO":
    if prompt_usuario := st.chat_input("Faça uma pergunta sobre a sua estratégia..."):
        st.session_state.historico_mensagens.append({"role": "user", "content": prompt_usuario})
        with st.chat_message("user"): st.markdown(prompt_usuario)
        with st.chat_message("assistant"):
            with st.spinner("A pensar..."):
                try:
                    response = st.session_state.agent_executor.invoke({"input": prompt_usuario})
                    resposta = response["output"]
                    st.markdown(resposta)
                except Exception as e:
                    resposta = f"Desculpe, ocorreu um erro: {e}"
                    st.error(resposta)
        st.session_state.historico_mensagens.append({"role": "assistant", "content": resposta})