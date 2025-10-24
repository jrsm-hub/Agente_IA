import os
import streamlit as st
from dotenv import load_dotenv
import json
import nest_asyncio
import re

# Importações do Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# Importações do LangChain
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from fpdf import FPDF, HTMLMixin
import markdown

#CLASSE PARA O PDF INTERPRETAR HTML
class PDF(FPDF, HTMLMixin):
    pass

#CONFIGURAÇÃO E CARREGAMENTO DE RECURSOS
nest_asyncio.apply()
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

#LIGAÇÃO AO FIREBASE
@st.cache_resource
def init_firebase():
    try:
        creds_json = st.secrets["firebase_credentials"]
        cred_dict = dict(creds_json)
        cred = credentials.Certificate(cred_dict)
    except (FileNotFoundError, KeyError):
        if not os.path.exists("firebase-credentials.json"):
            st.error("Ficheiro de credenciais do Firebase (firebase-credentials.json) não encontrado!")
            st.stop()
        cred = credentials.Certificate("firebase-credentials.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firebase()

#CARREGAMENTO DOS RECURSOS DE IA
@st.cache_resource
def carregar_recursos_ia():
    if not GOOGLE_API_KEY: st.error("Chave de API do Google não encontrada!"); st.stop()
    print("A inicializar os recursos de IA...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.8)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vectordb = Chroma(persist_directory="banco_vetorial_chroma", embedding_function=embeddings)
    print("Recursos de IA inicializados.")
    return llm, vectordb

llm, vectordb = carregar_recursos_ia()

#INICIALIZAÇÃO DO ESTADO DA SESSÃO 
def inicializar_sessao():
    st.session_state.fase = "INICIO"
    st.session_state.historico_mensagens = []
    st.session_state.agent_executor = None
    st.session_state.documento_gerado = None
    st.session_state.conversa_id = None

if "fase" not in st.session_state:
    inicializar_sessao()

#FUNÇÃO PARA GERAR TÍTULOS INTELIGENTES
def atualizar_titulo_conversa(conversa_id, historico_texto):
    """Usa o LLM para criar um título curto para a conversa e atualiza-o no Firebase."""
    template_titulo = "Com base no seguinte histórico de conversa, crie um título curto e descritivo (máximo 5 palavras) para este chat. Responda apenas com o título.\n\nHISTÓRICO:\n{historico}\n\nTÍTULO:"
    prompt = PromptTemplate.from_template(template_titulo)
    chain = prompt | llm
    try:
        novo_titulo = chain.invoke({"historico": historico_texto}).content
        # Atualiza o título no Firebase
        db.collection('conversas').document(conversa_id).update({'title': novo_titulo})
    except Exception as e:
        print(f"Erro ao atualizar o título da conversa: {e}")
        
#FUNÇÕES DE LÓGICA 
def gerar_proxima_pergunta_dinamica(historico_texto):
    template = """Você é um orientador de pesquisa experiente e a sua missão é conduzir uma entrevista para ajudar um aluno a definir um projeto de pesquisa, guiando-o desde uma área ampla até um tópico específico.

**Regras:**
1.  **Analise o histórico completo da conversa** para entender o contexto atual e a última resposta do aluno.
2.  **Formule a próxima pergunta** de forma clara, em português do Brasil. A sua pergunta deve ter o objetivo de afunilar a ideia do aluno.
3.  **Ofereça exatamente 4 opções** concretas e detalhadas de aprofundamento. Estas opções devem ser lógicas com base no que já foi discutido.
4.  **Formate as opções** como uma lista numerada, de 1 a 4, com cada opção numa nova linha.
5.  **Responda apenas com a pergunta e as opções**, sem qualquer texto adicional antes ou depois.

**HISTÓRICO DA CONVERSA:**
{historico}

**PRÓXIMA PERGUNTA:**"""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    return chain.invoke({"historico": historico_texto}).content

def gerar_documento_estrategico(historico_texto):
    template = """Você é um orientador de pesquisa. Com base na entrevista no HISTÓRICO, crie um documento de estratégia em Markdown e em português do Brasil.\nHISTÓRICO: {historico}\nDOCUMENTO ESTRATÉGICO:
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
    chain = prompt | llm
    with st.spinner("O estrategista está a consolidar a nossa conversa..."):
        documento_final = chain.invoke({"historico": historico_texto}).content
        st.session_state.documento_gerado = documento_final

def inicializar_agente_de_dialogo():
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever())
    tools = [Tool(name="Consulta_Manuais_Pesquisa", func=rag_chain.invoke, description="Use para perguntas sobre metodologia, escrita, etc.")]
    prompt = hub.pull("hwchase17/react-chat")
    prompt.template = prompt.template.replace("You are a helpful assistant.", "Você é um assistente de pesquisa. **RESPONDA SEMPRE EM PORTUGUÊS DO BRASIL.**")
    agent = create_react_agent(llm, tools, prompt)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for msg in st.session_state.historico_mensagens:
        if msg["role"] == "user": memory.chat_memory.add_user_message(msg["content"])
        else: memory.chat_memory.add_ai_message(msg["content"])
    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)

class FerramentaRAGComLogging:
    def __init__(self, llm, retriever):
        self.rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    def run(self, query: str) -> str:
        """Executa a consulta RAG e levanta a bandeira de consulta."""
        print(f"--- FERRAMENTA 'Consulta_Manuais_Pesquisa' ACIONADA COM A QUERY: {query} ---")
        
        
        st.session_state.manual_consultado = True
        
        return self.rag_chain.run(query)

#FUNÇÃO PARA CRIAR O PDF
def criar_pdf_formatado(texto_markdown):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    for linha in texto_markdown.split('\n'):
        linha_sanitizada = linha.encode('latin-1', 'replace').decode('latin-1')
        
        if linha.strip().startswith('### '):
            pdf.set_font("Arial", 'B', 13) # Define a fonte para negrito, tamanho 13
            pdf.multi_cell(0, 9, txt=linha_sanitizada.strip()[4:])
            pdf.ln(1)
            pdf.set_font("Arial", '', 12) # Volta para a fonte normal
        elif linha.strip().startswith('## '):
            pdf.set_font("Arial", 'B', 14)
            pdf.multi_cell(0, 10, txt=linha_sanitizada.strip()[3:])
            pdf.ln(2)
            pdf.set_font("Arial", '', 12)
        elif linha.strip().startswith('# '):
            pdf.set_font("Arial", 'B', 16)
            pdf.multi_cell(0, 12, txt=linha_sanitizada.strip()[2:])
            pdf.ln(4)
            pdf.set_font("Arial", '', 12)
        elif linha.strip().startswith('* '):
            pdf.cell(5) # Indentação
            # A lógica de negrito será aplicada dentro desta linha se necessário
            segmentos = re.split(r'(\*\*.*?\*\*)', linha_sanitizada.strip()[2:])
            for segmento in segmentos:
                if segmento.startswith('**') and segmento.endswith('**'):
                    pdf.set_font('Arial', 'B')
                    pdf.write(7, segmento[2:-2])
                    pdf.set_font('Arial', '')
                else:
                    pdf.write(7, segmento)
            pdf.ln(7)
        else:
            # Lógica para tratar o negrito em linhas normais
            segmentos = re.split(r'(\*\*.*?\*\*)', linha_sanitizada)
            
            for segmento in segmentos:
                if segmento.startswith('**') and segmento.endswith('**'):
                    pdf.set_font('Arial', 'B')
                    pdf.write(7, segmento[2:-2])
                    pdf.set_font('Arial', '')
                else:
                    pdf.write(7, segmento)
            
            pdf.ln(7) # Pula para a próxima linha
            if not linha.strip():
                pdf.ln(4)

    return pdf.output(dest='S').encode('latin-1')


#INTERFACE PRINCIPAL DO STREAMLIT
st.set_page_config(page_title="Estrategista de Pesquisa", page_icon="🧭")
st.title("🧭 Estrategista de Pesquisa Académica")

with st.sidebar:
    st.header("Minhas Conversas")
    
    if st.button("➕ Nova Conversa"):
        inicializar_sessao()
        
        nova_conversa_ref = db.collection('conversas').document()
        st.session_state.conversa_id = nova_conversa_ref.id
        st.session_state.fase = "COLETA"
        
        mensagem_inicial = "Olá! Sou o seu estrategista de pesquisa. Vamos construir juntos a base para o seu trabalho académico. Para começar, qual é a sua grande área de interesse?"
        st.session_state.historico_mensagens = [{"role": "assistant", "content": mensagem_inicial}]
        
        nova_conversa_ref.set({'timestamp': firestore.SERVER_TIMESTAMP, 'title': f'Conversa {nova_conversa_ref.id[:4]}...'})
        nova_conversa_ref.collection('mensagens').add({'role': 'assistant', 'content': mensagem_inicial, 'timestamp': firestore.SERVER_TIMESTAMP})
        st.rerun()

    conversas = db.collection('conversas').order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
    for conversa in conversas:
        if st.button(conversa.to_dict().get('title', 'Conversa Antiga'), key=conversa.id):
            inicializar_sessao()
            st.session_state.conversa_id = conversa.id
            msgs_db = db.collection('conversas').document(conversa.id).collection('mensagens').order_by('timestamp').stream()
            st.session_state.historico_mensagens = [msg.to_dict() for msg in msgs_db]
            doc_gerado = any("Estratégia de Pesquisa" in msg.get('content', '') for msg in st.session_state.historico_mensagens)
            if doc_gerado:
                st.session_state.fase = "DIALOGO_ABERTO"
                st.session_state.documento_gerado = next((msg['content'] for msg in st.session_state.historico_mensagens if "Estratégia de Pesquisa" in msg.get('content', '')), None)
            else:
                st.session_state.fase = "COLETA"
            st.rerun()

# Exibe o histórico
if "historico_mensagens" in st.session_state:
    for msg in st.session_state.historico_mensagens:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

#LÓGICA DAS FASES
if st.session_state.fase == "INICIO":
    st.info("Bem-vindo! Selecione 'Nova Conversa' na barra lateral para começar a estruturar a sua ideia de pesquisa.")

elif st.session_state.fase == "COLETA":
    if not st.session_state.get("conversa_id"):
        st.session_state.fase = "INICIO"
        st.rerun()
    if len(st.session_state.historico_mensagens) > 0:
        if st.button("Já temos informação suficiente. Gerar Documento Estratégico!"):
            st.session_state.fase = "GERACAO"
            st.rerun()
    if prompt_usuario := st.chat_input("Sua resposta..."):
        st.session_state.historico_mensagens.append({"role": "user", "content": prompt_usuario})
        db.collection('conversas').document(st.session_state.conversa_id).collection('mensagens').add({'role': 'user', 'content': prompt_usuario, 'timestamp': firestore.SERVER_TIMESTAMP})
        
        historico_texto = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.historico_mensagens])
         # CHAMADA À NOVA FUNÇÃO DE TÍTULO
        # Se for a primeira resposta do utilizador, atualiza o título
        if len(st.session_state.historico_mensagens) == 2: # (msg assistente + 1ª msg user)
            atualizar_titulo_conversa(st.session_state.conversa_id, historico_texto)

        with st.chat_message("assistant"):
            with st.spinner("Formulando a próxima pergunta..."):
                proxima_pergunta = gerar_proxima_pergunta_dinamica(historico_texto)
                st.markdown(proxima_pergunta)
                st.session_state.historico_mensagens.append({"role": "assistant", "content": proxima_pergunta})
                db.collection('conversas').document(st.session_state.conversa_id).collection('mensagens').add({'role': 'assistant', 'content': proxima_pergunta, 'timestamp': firestore.SERVER_TIMESTAMP})
        st.rerun()
elif st.session_state.fase == "GERACAO":
    if st.session_state.documento_gerado is None:
        historico_texto = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.historico_mensagens])
        gerar_documento_estrategico(historico_texto)
        st.session_state.historico_mensagens.append({"role": "assistant", "content": st.session_state.documento_gerado})
        db.collection('conversas').document(st.session_state.conversa_id).collection('mensagens').add({'role': 'assistant', 'content': st.session_state.documento_gerado, 'timestamp': firestore.SERVER_TIMESTAMP})
        
        # LÓGICA DE SCROLL 
        # Define uma "bandeira" para indicar que o scroll deve acontecer na próxima recarga.
        st.session_state.scroll_to_bottom = True
        st.rerun()
    else:
        # NOVA LÓGICA DE SCROLL 
        # Verifica se a "bandeira" de scroll está ativa.
        if st.session_state.get("scroll_to_bottom"):
            # Injeta o código JavaScript para rolar até o final da página.
            js = "<script>window.scrollTo(0, document.body.scrollHeight);</script>"
            st.html(js)
            # Desativa a "bandeira" para que o scroll não aconteça repetidamente.
            st.session_state.scroll_to_bottom = False

        pdf_bytes = criar_pdf_formatado(st.session_state.documento_gerado)
        st.download_button(
            label="Descarregar Estratégia em PDF",
            data=pdf_bytes,
            file_name="estrategia_de_pesquisa.pdf",
            mime="application/pdf"
        )
        
        if st.button("Excelente! Agora vamos discutir esta estratégia"):
            st.session_state.fase = "DIALOGO_ABERTO"
            st.session_state.agent_executor = inicializar_agente_de_dialogo()
            msg_transicao = "A sua estratégia de pesquisa está acima. Agora, estou pronto para discutir e aprofundar os pontos. O que gostaria de explorar primeiro?"
            st.session_state.historico_mensagens.append({"role": "assistant", "content": msg_transicao})
            db.collection('conversas').document(st.session_state.conversa_id).collection('mensagens').add({'role': 'assistant', 'content': msg_transicao, 'timestamp': firestore.SERVER_TIMESTAMP})
            st.rerun()
            
elif st.session_state.fase == "DIALOGO_ABERTO":
    if prompt_usuario := st.chat_input("Faça uma pergunta sobre a sua estratégia..."):
        st.session_state.historico_mensagens.append({"role": "user", "content": prompt_usuario})
        db.collection('conversas').document(st.session_state.conversa_id).collection('mensagens').add({'role': 'user', 'content': prompt_usuario, 'timestamp': firestore.SERVER_TIMESTAMP})
        
        with st.chat_message("assistant"):
            with st.spinner("A pensar..."):
                try:
                    if not st.session_state.get('agent_executor'):
                         st.session_state.agent_executor = inicializar_agente_de_dialogo()
                    
                    response = st.session_state.agent_executor.invoke({"input": prompt_usuario})
                    resposta = response["output"]
                    st.markdown(resposta)
                except Exception as e:
                    resposta = f"Desculpe, ocorreu um erro: {e}"
                    st.error(resposta)
        st.session_state.historico_mensagens.append({"role": "assistant", "content": resposta})
        db.collection('conversas').document(st.session_state.conversa_id).collection('mensagens').add({'role': 'assistant', 'content': resposta, 'timestamp': firestore.SERVER_TIMESTAMP})