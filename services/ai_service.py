import os
import re
import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from services.rag_service import criar_rag_tool

def separar_pergunta_e_opcoes(texto_mensagem: str) -> tuple[str, list[str]]:
    """Separa o texto introdutório da pergunta das opções numeradas (1 a 4), evitando duplicação visual."""
    if not texto_mensagem:
        return "", []
    
    linhas = texto_mensagem.strip().split('\n')
    linhas_pergunta = []
    opcoes = []
    
    for linha in linhas:
        linha_strip = linha.strip()
        match = re.match(r'^\*{0,2}([1-4])[\.\)\-\:]\s*(.+?)\*{0,2}$', linha_strip)
        if match:
            opcao_texto = match.group(2).strip()
            opcao_texto = re.sub(r'^\*\*(.*?)\*\*$', r'\1', opcao_texto).strip()
            if opcao_texto:
                opcoes.append(opcao_texto)
        else:
            if not opcoes:  # Todas as linhas antes de começar a listagem de opções
                linhas_pergunta.append(linha)
                
    pergunta_limpa = "\n".join(linhas_pergunta).strip()
    return pergunta_limpa, opcoes

def extrair_opcoes_da_pergunta(texto_mensagem: str) -> list[str]:
    """Retorna apenas as opções numeradas extraídas da mensagem."""
    _, opcoes = separar_pergunta_e_opcoes(texto_mensagem)
    return opcoes

@st.cache_resource
def carregar_llm(temperature: float = LLM_TEMPERATURE):
    """Inicializa o modelo ChatGoogleGenerativeAI padrão para diálogos criativos."""
    if not GOOGLE_API_KEY:
        st.error("Chave de API do Google (GOOGLE_API_KEY) não encontrada!")
        st.stop()
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature
    )

@st.cache_resource
def carregar_llm_estrito():
    """Inicializa o modelo ChatGoogleGenerativeAI com temperatura zero para máxima fidelidade científica (anti-alucinação)."""
    if not GOOGLE_API_KEY:
        st.error("Chave de API do Google (GOOGLE_API_KEY) não encontrada!")
        st.stop()
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0
    )


def gerar_titulo_conversa(llm, historico_texto: str) -> str:
    """Usa o LLM para criar um título curto (máx 5 palavras) para a conversa."""
    template_titulo = (
        "Com base no seguinte histórico de conversa, crie um título curto e descritivo "
        "(máximo 5 palavras) para este chat. Responda apenas com o título.\n\n"
        "HISTÓRICO:\n{historico}\n\nTÍTULO:"
    )
    prompt = PromptTemplate.from_template(template_titulo)
    chain = prompt | llm
    try:
        resultado = chain.invoke({"historico": historico_texto})
        return resultado.content.strip().replace('"', '')
    except Exception as e:
        print(f"Erro ao gerar título: {e}")
        return ""

def gerar_proxima_pergunta_dinamica(llm, historico_texto: str) -> str:
    """Gera a próxima pergunta afuniladora e 4 opções para a fase de coleta (síncrono)."""
    template = """Você é um orientador de pesquisa experiente e a sua missão é conduzir uma entrevista para ajudar um aluno a definir um projeto de pesquisa, guiando-o desde uma área ampla até um tópico específico.

**Regras:**
1. **Analise o histórico completo da conversa** para entender o contexto atual e a última resposta do aluno.
2. **Formule a próxima pergunta** de forma clara, em português do Brasil. A sua pergunta deve ter o objetivo de afunilar a ideia do aluno.
3. **Ofereça exatamente 4 opções** concretas e detalhadas de aprofundamento. Estas opções devem ser lógicas com base no que já foi discutido.
4. **Formate as opções** como uma lista numerada, de 1 a 4, com cada opção numa nova linha.
5. **Responda apenas com a pergunta e as opções**, sem qualquer texto adicional antes ou depois.

**HISTÓRICO DA CONVERSA:**
{historico}

**PRÓXIMA PERGUNTA:**"""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    return chain.invoke({"historico": historico_texto}).content

def stream_gerar_proxima_pergunta(llm, historico_texto: str):
    """Gera a próxima pergunta afuniladora e 4 opções em formato de streaming."""
    template = """Você é um orientador de pesquisa experiente e a sua missão é conduzir uma entrevista para ajudar um aluno a definir um projeto de pesquisa, guiando-o desde uma área ampla até um tópico específico.

**Regras:**
1. **Analise o histórico completo da conversa** para entender o contexto atual e a última resposta do aluno.
2. **Formule a próxima pergunta** de forma clara, em português do Brasil. A sua pergunta deve ter o objetivo de afunilar a ideia do aluno.
3. **Ofereça exatamente 4 opções** concretas e detalhadas de aprofundamento. Estas opções devem ser lógicas com base no que já foi discutido.
4. **Formate as opções** como uma lista numerada, de 1 a 4, com cada opção numa nova linha.
5. **Responda apenas com a pergunta e as opções**, sem qualquer texto adicional antes ou depois.

**HISTÓRICO DA CONVERSA:**
{historico}

**PRÓXIMA PERGUNTA:**"""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    for chunk in chain.stream({"historico": historico_texto}):
        if hasattr(chunk, 'content'):
            yield chunk.content
        else:
            yield str(chunk)

def gerar_documento_estrategico(llm, historico_texto: str) -> str:
    """Gera o documento de estratégia de pesquisa em Markdown a partir do histórico (síncrono)."""
    template = """Você é um orientador de pesquisa. Com base na entrevista no HISTÓRICO, crie um documento de estratégia em Markdown e em português do Brasil.
HISTÓRICO: {historico}
DOCUMENTO ESTRATÉGICO:
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
    return chain.invoke({"historico": historico_texto}).content

def stream_gerar_documento_estrategico(llm, historico_texto: str):
    """Gera o documento de estratégia de pesquisa em formato de streaming."""
    template = """Você é um orientador de pesquisa. Com base na entrevista no HISTÓRICO, crie um documento de estratégia em Markdown e em português do Brasil.
HISTÓRICO: {historico}
DOCUMENTO ESTRATÉGICO:
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
    for chunk in chain.stream({"historico": historico_texto}):
        if hasattr(chunk, 'content'):
            yield chunk.content
        else:
            yield str(chunk)

def inicializar_agente_de_dialogo(llm, vectordb, historico_mensagens: list) -> AgentExecutor:
    """Inicializa o agente ReAct para diálogo livre com acesso às ferramentas de RAG."""
    rag_tool = criar_rag_tool(vectordb, llm)
    tools = [rag_tool]
    
    prompt_agente = hub.pull("hwchase17/react-chat")

    nova_instrucao_dialogo = """Você é um assistente de pesquisa e a sua missão é continuar uma conversa com um aluno para ajudá-lo a desenvolver sua pesquisa.

Regras Importantes:
1. **Contexto:** O histórico da conversa contém a entrevista inicial e um documento estratégico que você já forneceu. Use esse contexto para guiar suas respostas.
2. **Honestidade:** Se você não souber a resposta para uma pergunta ou não tiver certeza, é crucial que você responda honestamente que não sabe ou não tem certeza. **NÃO INVENTE INFORMAÇÕES.**
3. **Idioma:** Responda sempre em **PORTUGUÊS DO BRASIL**.
"""
    
    prompt_agente.template = prompt_agente.template.replace(
        "You are a helpful assistant. Respond to the user's request as best you can.",
        nova_instrucao_dialogo
    ).replace("Begin!", "Comece!").replace("Thought:", "Pensamento:").replace("Action:", "Ação:").replace("Action Input:", "Entrada da Ação:").replace("Observation:", "Observação:")

    agent = create_react_agent(llm, tools, prompt_agente)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    for msg in historico_mensagens:
        if msg.get("role") == "user":
            memory.chat_memory.add_user_message(msg.get("content", ""))
        else:
            memory.chat_memory.add_ai_message(msg.get("content", ""))
            
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

def stream_resposta_rag_estrito(llm, vectordb, pergunta: str, historico_mensagens: list):
    """Executa a busca RAG no ChromaDB e gera resposta em streaming com rigor científico e citação de fontes."""
    # Recupera os 4 documentos mais relevantes do banco vetorial
    docs = vectordb.similarity_search(pergunta, k=4)
    
    if docs:
        contexto = "\n\n".join([
            f"--- TRECHO DO MANUAL ({os.path.basename(doc.metadata.get('source', 'Manual Desconhecido'))}) ---\n{doc.page_content}"
            for doc in docs
        ])
    else:
        contexto = "Nenhum trecho relevante foi recuperado da base de dados."

    # Prepara histórico resumido recente
    msgs_recentes = historico_mensagens[-6:] if len(historico_mensagens) > 6 else historico_mensagens
    historico_str = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs_recentes])

    template_estrito = """Você é um Consultor e Metodólogo Científico sênior, com rigor absoluto e fidelidade total à literatura acadêmica.

Sua missão é responder à dúvida metodológica do aluno com base ESTRITAMENTE nos fragmentos de manuais e livros fornecidos abaixo.

DIRETRIZES DE FIDELIDADE (ANTI-ALUCINAÇÃO):
1. Use SOMENTE os fatos, conceitos, normas e métodos presentes nos TRECHOS DA LITERATURA CIENTÍFICA fornecidos. NÃO invente informações ou regras que não constem no texto.
2. Se a informação não constar nos trechos fornecidos, declare explicitamente: "Com base na literatura acadêmica carregada no sistema, não foi possível encontrar informações suficientes para responder a este ponto específico com exatidão." e recomende qual tipo de literatura buscar.
3. Seja didático, claro, formal e encorajador, sempre em PORTUGUÊS DO BRASIL.

TRECHOS DA LITERATURA CIENTÍFICA:
{contexto}

HISTÓRICO DA CONVERSA:
{historico}

DÚVIDA DO ALUNO:
{pergunta}

RESPOSTA CIENTÍFICA FUNDAMENTADA:"""

    prompt = PromptTemplate.from_template(template_estrito)
    chain = prompt | llm

    for chunk in chain.stream({"contexto": contexto, "historico": historico_str, "pergunta": pergunta}):
        if hasattr(chunk, 'content'):
            yield chunk.content
        else:
            yield str(chunk)
            
    # Adiciona a citação formatada das fontes encontradas
    fontes_unicas = sorted(list(set(
        os.path.basename(doc.metadata.get('source', 'Manual'))
        for doc in docs if doc.metadata.get('source')
    )))
    if fontes_unicas:
        fontes_md = "\n\n---\n📚 **Obras Consultadas na Base:**\n" + "\n".join([f"- *{f.replace('.txt', '')}*" for f in fontes_unicas])
        yield fontes_md

