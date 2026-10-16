import os
import re
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE

def extrair_texto_conteudo(conteudo) -> str:
    """Garante a conversão segura de qualquer formato retornado pelo LLM (str, list, dict, AIMessageChunk) em texto string."""
    if conteudo is None:
        return ""
    if isinstance(conteudo, str):
        return conteudo
    if isinstance(conteudo, list):
        partes = []
        for item in conteudo:
            if isinstance(item, str):
                partes.append(item)
            elif isinstance(item, dict):
                txt = item.get("text") or item.get("content") or item.get("thought") or ""
                partes.append(str(txt))
            elif hasattr(item, "text"):
                partes.append(str(item.text))
            elif hasattr(item, "content"):
                partes.append(extrair_texto_conteudo(item.content))
            else:
                partes.append(str(item))
        return "".join(partes)
    if isinstance(conteudo, dict):
        txt = conteudo.get("text") or conteudo.get("content") or ""
        return str(txt) if txt else str(conteudo)
    if hasattr(conteudo, "content"):
        return extrair_texto_conteudo(conteudo.content)
    return str(conteudo)

def separar_pergunta_e_opcoes(texto_mensagem) -> tuple[str, list[str]]:
    """Separa o texto introdutório da pergunta das opções numeradas (1 a 4), preservando integralmente o texto."""
    texto_str = extrair_texto_conteudo(texto_mensagem)
    if not texto_str:
        return "", []
    
    linhas = texto_str.strip().split('\n')
    linhas_pergunta = []
    opcoes = []
    
    for linha in linhas:
        linha_strip = linha.strip()
        if not linha_strip:
            if not opcoes:
                linhas_pergunta.append(linha)
            continue
            
        # Trata linhas com negrito externo total (ex: **1. Opção**)
        linha_processar = linha_strip
        if linha_processar.startswith("**") and linha_processar.endswith("**") and len(linha_processar) > 4:
            linha_processar = linha_processar[2:-2].strip()
            
        match = re.match(r'^[1-4][\.\)\-\:]\s*(.+)$', linha_processar)
        if match:
            opcao_texto = match.group(1).strip()
            if opcao_texto:
                opcoes.append(opcao_texto)
        else:
            if not opcoes:  # Todas as linhas antes de começar a listagem de opções
                linhas_pergunta.append(linha)
            elif linha_strip and not re.match(r'^(Observação|Nota|Escolha|Selecione|Ou digite)', linha_strip, re.I):
                # Linha de continuação da opção anterior caso o modelo quebre a linha
                opcoes[-1] = f"{opcoes[-1]} {linha_strip}"
                
    pergunta_limpa = "\n".join(linhas_pergunta).strip()
    return pergunta_limpa, opcoes

def extrair_opcoes_da_pergunta(texto_mensagem) -> list[str]:
    """Retorna apenas as opções numeradas extraídas da mensagem."""
    _, opcoes = separar_pergunta_e_opcoes(texto_mensagem)
    return opcoes

def assegurar_modelo_valido(llm):
    """Garante que qualquer instância de ChatGoogleGenerativeAI use gemini-3.5-flash-lite caso esteja com modelo descontinuado ou sem cota."""
    if hasattr(llm, "model") and llm.model:
        nome = str(llm.model)
        if "2.5" in nome or "1.5" in nome or "3.8" in nome or nome in ["gemini-2.5-flash", "models/gemini-2.5-flash", "gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-3.8-flash"]:
            llm.model = "gemini-3.5-flash-lite"
    return llm

@st.cache_resource
def carregar_llm(api_key: str = None, temperature: float = LLM_TEMPERATURE, model_name: str = None):
    """Inicializa o modelo ChatGoogleGenerativeAI padrão para diálogos acadêmicos."""
    chave = (api_key or GOOGLE_API_KEY or "").strip()
    if not chave:
        st.error("Chave de API do Google (GOOGLE_API_KEY) não encontrada!")
        st.stop()
    modelo = model_name or LLM_MODEL or "gemini-3.5-flash-lite"
    if "2.5" in str(modelo) or "1.5" in str(modelo) or "3.8" in str(modelo):
        modelo = "gemini-3.5-flash-lite"
    return ChatGoogleGenerativeAI(
        model=modelo,
        google_api_key=chave,
        temperature=temperature
    )

@st.cache_resource
def carregar_llm_estrito(api_key: str = None, model_name: str = None):
    """Inicializa o modelo ChatGoogleGenerativeAI com temperatura zero para máxima fidelidade científica."""
    chave = (api_key or GOOGLE_API_KEY or "").strip()
    if not chave:
        st.error("Chave de API do Google (GOOGLE_API_KEY) não encontrada!")
        st.stop()
    modelo = model_name or LLM_MODEL or "gemini-3.5-flash-lite"
    if "2.5" in str(modelo) or "1.5" in str(modelo) or "3.8" in str(modelo):
        modelo = "gemini-3.5-flash-lite"
    return ChatGoogleGenerativeAI(
        model=modelo,
        google_api_key=chave,
        temperature=0.0
    )


def extrair_tema_documento_estrategico(documento_texto: str) -> str:
    """Extrai o tema delimitado ou subtópico de um Documento Estratégico em Markdown."""
    if not documento_texto:
        return ""
    
    padroes = [
        r'[\-\*]?\s*\*\*(?:Subárea\s*/\s*)?(?:Tópico\s*Delimitado|Tema\s*Delimitado|Delimitação\s*Temática)\s*:\*\*\s*(.+)',
        r'###\s*Caminho\s*A[^\:\n]*\:\s*(.+)',
        r'[\-\*]?\s*\*\*Grande\s*Área\s*:\*\*\s*(.+)'
    ]
    
    for padrao in padroes:
        match = re.search(padrao, documento_texto, re.IGNORECASE)
        if match:
            tema = match.group(1).strip()
            # Remove marcadores de exemplo [exemplo]
            tema = re.sub(r'\[(.*?)\]', r'\1', tema).strip()
            # Remove markdown e aspas
            tema = tema.replace("*", "").replace('"', '').replace("'", "").replace("`", "").strip()
            if tema and not tema.startswith("[") and not tema.startswith("<"):
                palavras = tema.split()
                return " ".join(palavras[:6]) if len(palavras) > 6 else tema

    return ""

def gerar_titulo_conversa(llm, historico_texto: str) -> str:
    """Usa o LLM para identificar o tema de pesquisa e criar um título curto (máx 5 palavras)."""
    template_titulo = (
        "Você é um orientador acadêmico. Com base no diálogo a seguir, identifique o TEMA DE PESQUISA "
        "ou área de investigação delimitada pelo aluno e crie um título curto e formal (entre 2 a 5 palavras). "
        "Exemplos de bons títulos: 'Visão Computacional no Glaucoma', 'LLMs em Prontuários Médicos', "
        "'Otimização de Redes com GNNs'.\n"
        "Regras: Não use emojis, não use aspas, não escreva 'Título:', responda APENAS com o nome do tema de pesquisa.\n\n"
        "HISTÓRICO:\n{historico}\n\nTEMA DE PESQUISA:"
    )
    prompt = PromptTemplate.from_template(template_titulo)
    chain = prompt | llm
    try:
        resultado = chain.invoke({"historico": historico_texto})
        texto = extrair_texto_conteudo(resultado)
        texto = texto.strip().replace('"', '').replace("'", "").replace("`", "")
        # Remove prefixos indesejados caso o modelo responda 'Título: ...' ou 'Tema: ...'
        texto = re.sub(r'^(T[íi]tulo|Tema|Tema de Pesquisa|Assunto)\s*:\s*', '', texto, flags=re.IGNORECASE).strip()
        linhas = [l.strip() for l in texto.split('\n') if l.strip()]
        return linhas[0] if linhas else texto
    except Exception as e:
        print(f"Erro ao gerar título: {e}")
        return ""

def gerar_proxima_pergunta_dinamica(llm, historico_texto: str) -> str:
    """Gera a próxima pergunta afuniladora e 4 opções para a fase de coleta (síncrono)."""
    llm = assegurar_modelo_valido(llm)
    template = """Você é um orientador de pesquisa experiente e a sua missão é conduzir uma entrevista para ajudar um aluno a definir um projeto de pesquisa, guiando-o desde uma área ampla até um tópico específico.

**Regras:**
1. **Analise o histórico completo da conversa** para entender o contexto atual e a última resposta do aluno.
2. **Formule a próxima pergunta** de forma clara e rigorosa, em português do Brasil. A sua pergunta deve afunilar a ideia do aluno.
3. **Ofereça exatamente 4 opções** concretas, fundamentadas e detalhadas de aprofundamento.
4. **Formate as opções** como uma lista numerada, de 1 a 4, com cada opção numa nova linha.
5. **Estilo sóbrio:** Não use emojis, ícones gráficos ou expressões informais.
6. **Responda apenas com a pergunta e as opções**, sem qualquer texto adicional antes ou depois.

**HISTÓRICO DA CONVERSA:**
{historico}

**PRÓXIMA PERGUNTA:**"""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    try:
        resultado = chain.invoke({"historico": historico_texto})
        return extrair_texto_conteudo(resultado)
    except Exception as e:
        if "404" in str(e) or "NOT_FOUND" in str(e) or "2.5" in str(e) or "429" in str(e) or "503" in str(e):
            llm.model = "gemini-3.5-flash-lite"
            resultado = (prompt | llm).invoke({"historico": historico_texto})
            return extrair_texto_conteudo(resultado)
        raise e

def stream_gerar_proxima_pergunta(llm, historico_texto: str, vectordb=None):
    """Gera a próxima pergunta afuniladora e 4 opções em streaming com alta responsividade."""
    llm = assegurar_modelo_valido(llm)
    template = """Você é um orientador de pesquisa experiente e a sua missão é conduzir uma entrevista socrática para ajudar um aluno a delimitar seu projeto de pesquisa acadêmica, guiando-o desde uma área ampla até um tópico específico, rigoroso e exequível.

**Regras:**
1. **Analise o histórico completo da conversa** para entender o contexto atual e a última contribuição do aluno.
2. **Formule a próxima pergunta** de forma clara, reflexiva e objetiva em português do Brasil, estimulando o afunilamento temático.
3. **Ofereça exatamente 4 opções** metodologicamente consistentes, inovadoras e detalhadas de aprofundamento.
4. **Formate as opções** rigorosamente como lista numerada de 1 a 4, cada qual em uma linha separada.
5. **Soberba acadêmica:** Proibido utilizar emojis, ícones figurativos, gírias ou pontuações lúdicas. Mantenha tom formal e institucional de orientação universitária.
6. **Responda estritamente com a pergunta seguida das 4 opções**, sem saudações redundantes ou introduções.

**HISTÓRICO DA CONVERSA:**
{historico}

**PRÓXIMA PERGUNTA:**"""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    try:
        for chunk in chain.stream({"historico": historico_texto}):
            txt = extrair_texto_conteudo(chunk)
            if txt:
                yield txt
    except Exception as e:
        err_str = str(e)
        if "404" in err_str or "NOT_FOUND" in err_str or "2.5" in err_str or "available" in err_str or "429" in err_str or "503" in err_str:
            llm.model = "gemini-3.5-flash-lite"
            chain_retry = prompt | llm
            for chunk in chain_retry.stream({"historico": historico_texto}):
                txt = extrair_texto_conteudo(chunk)
                if txt:
                    yield txt
        else:
            raise e

TEMPLATE_DOCUMENTO_ESTRATEGICO = """Você é um orientador e estrategista acadêmico sênior. 
Com base em todo o HISTÓRICO da entrevista com o aluno, elabore um Documento Estratégico de Pesquisa detalhado, aprofundado e completo em Markdown (em português do Brasil).

Não seja superficial. Desenvolva cada seção com rigor conceitual, exemplos práticos e clareza metodológica.

HISTÓRICO DA ENTREVISTA:
{historico}

DOCUMENTO ESTRATÉGICO DE PESQUISA (ESTRUTURA OBRIGATÓRIA):

# Estratégia de Pesquisa Acadêmica

## 1. Diagnóstico do Perfil e Delimitação Temática
- **Grande Área:** [Área identificada]
- **Subárea / Tópico Delimitado:** [Delimitação resultante do afunilamento]
- **Problema de Pesquisa:** [Descrição aprofundada do problema e da dor prática/teórica que o trabalho busca resolver]
- **Pergunta Norteadora:** [A pergunta de pesquisa central em formato formal de interrogação]

## 2. Caminhos Metodológicos Propostos

### Caminho A (Abordagem Principal): [Nome Detalhado da Abordagem Principal]
- **Descrição da Proposta:** [Como funciona esta abordagem, arquitetura ou técnica pretendida]
- **Objetivo Geral & Específicos:** [O que será construído/analisado passo a passo]
- **Metodologia & Dados:** [Quais bases de dados, ferramentas, métricas ou procedimentos serão adotados]
- **Por que este caminho é promissor:** [Contribuição científica e relevância]
- **Riscos e Mitigações:** [Principais desafios técnicos/metodológicos e como contorná-los]

### Caminho B (Abordagem Alternativa): [Nome Detalhado da Abordagem Alternativa]
- **Descrição da Proposta:** [Abordagem alternativa para o mesmo tema com foco ou escopo diferente]
- **Objetivo Geral & Específicos:** [O que será construído/analisado passo a passo]
- **Metodologia & Dados:** [Quais bases de dados, ferramentas, métricas ou procedimentos serão adotados]
- **Por que este caminho é promissor:** [Diferencial em relação ao Caminho A]
- **Riscos e Mitigações:** [Principais desafios técnicos/metodológicos e como contorná-los]

## 3. Recomendações Metodológicas e Próximos Passos
1. **Passo 1 (Revisão Bibliográfica):** [Direcionamento de busca de artigos e autores-chave]
2. **Passo 2 (Preparação do Ambiente / Dados):** [Coleta, ferramentas e pipeline inicial]
3. **Passo 3 (Alinhamento com Orientador):** [Pontos centrais a apresentar para o orientador humano]
"""

def gerar_documento_estrategico(llm, historico_texto: str) -> str:
    """Gera o documento de estratégia de pesquisa em Markdown a partir do histórico (síncrono)."""
    llm = assegurar_modelo_valido(llm)
    prompt = PromptTemplate.from_template(TEMPLATE_DOCUMENTO_ESTRATEGICO)
    chain = prompt | llm
    try:
        resultado = chain.invoke({"historico": historico_texto})
        return extrair_texto_conteudo(resultado)
    except Exception as e:
        if "404" in str(e) or "NOT_FOUND" in str(e) or "2.5" in str(e) or "429" in str(e) or "503" in str(e):
            llm.model = "gemini-3.5-flash-lite"
            resultado = (prompt | llm).invoke({"historico": historico_texto})
            return extrair_texto_conteudo(resultado)
        raise e

def stream_gerar_documento_estrategico(llm, historico_texto: str):
    """Gera o documento de estratégia de pesquisa em formato de streaming."""
    llm = assegurar_modelo_valido(llm)
    prompt = PromptTemplate.from_template(TEMPLATE_DOCUMENTO_ESTRATEGICO)
    chain = prompt | llm
    try:
        for chunk in chain.stream({"historico": historico_texto}):
            txt = extrair_texto_conteudo(chunk)
            if txt:
                yield txt
    except Exception as e:
        if "404" in str(e) or "NOT_FOUND" in str(e) or "2.5" in str(e) or "429" in str(e) or "503" in str(e):
            llm.model = "gemini-3.5-flash-lite"
            for chunk in (prompt | llm).stream({"historico": historico_texto}):
                txt = extrair_texto_conteudo(chunk)
                if txt:
                    yield txt
        else:
            raise e

class DialogoAcademicoAgent:
    """Agente conversacional com memória contextual e consulta integrada aos manuais RAG."""
    def __init__(self, llm, vectordb, historico_mensagens: list):
        self.llm = assegurar_modelo_valido(llm)
        self.vectordb = vectordb
        self.historico_mensagens = historico_mensagens

    def invoke(self, inputs: dict, config: dict = None) -> dict:
        pergunta = inputs.get("input", "")
        
        # Consulta semântica aos manuais de pesquisa
        try:
            docs = self.vectordb.similarity_search(pergunta, k=3)
            contexto = "\n\n".join([
                f"[Manual: {os.path.basename(doc.metadata.get('source', 'Referência'))}]\n{doc.page_content}"
                for doc in docs
            ]) if docs else ""
        except Exception as e:
            print(f"Aviso na busca vetorial do diálogo: {e}")
            contexto = ""
            
        msgs_recentes = self.historico_mensagens[-8:] if len(self.historico_mensagens) > 8 else self.historico_mensagens
        historico_str = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs_recentes])
        
        prompt = (
            "Você é um orientador e estrategista acadêmico experiente em Ciência da Computação e Metodologia Científica.\n"
            "Sua missão é continuar a conversa com o aluno, tirando dúvidas sobre o projeto de pesquisa, aprofundando o tema e orientando os próximos passos.\n\n"
            f"HISTÓRICO RECENTE DA CONVERSA:\n{historico_str}\n\n"
            f"TRECHOS DE MANUAIS METODOLÓGICOS RELEVANTES:\n{contexto}\n\n"
            f"PERGUNTA DO ALUNO: {pergunta}\n\n"
            "Responda de forma didática, encorajadora, estruturada e em português do Brasil, fundamentando suas recomendações com rigor acadêmico."
        )
        
        try:
            resposta_raw = self.llm.invoke(prompt)
            resposta = extrair_texto_conteudo(resposta_raw)
        except Exception as e:
            resposta = f"Ocorreu um erro ao processar a resposta: {e}"
            
        return {"output": resposta}

    def stream(self, inputs: dict):
        """Gera resposta em formato streaming."""
        pergunta = inputs.get("input", "")
        
        # Consulta semântica aos manuais de pesquisa
        try:
            docs = self.vectordb.similarity_search(pergunta, k=3)
            contexto = "\n\n".join([
                f"[Manual: {os.path.basename(doc.metadata.get('source', 'Referência'))}]\n{doc.page_content}"
                for doc in docs
            ]) if docs else ""
        except Exception as e:
            print(f"Aviso na busca vetorial do diálogo: {e}")
            contexto = ""
            
        msgs_recentes = self.historico_mensagens[-8:] if len(self.historico_mensagens) > 8 else self.historico_mensagens
        historico_str = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs_recentes])
        
        prompt = (
            "Você é um orientador e estrategista acadêmico experiente em Ciência da Computação e Metodologia Científica.\n"
            "Sua missão é continuar a conversa com o aluno, tirando dúvidas sobre o projeto de pesquisa, aprofundando o tema e orientando os próximos passos.\n\n"
            f"HISTÓRICO RECENTE DA CONVERSA:\n{historico_str}\n\n"
            f"TRECHOS DE MANUAIS METODOLÓGICOS RELEVANTES:\n{contexto}\n\n"
            f"PERGUNTA DO ALUNO: {pergunta}\n\n"
            "Responda de forma didática, encorajadora, estruturada e em português do Brasil, fundamentando suas recomendações com rigor acadêmico."
        )
        
        for chunk in self.llm.stream(prompt):
            txt = extrair_texto_conteudo(chunk)
            if txt:
                yield txt

def stream_dialogo_academico(llm, vectordb, pergunta: str, historico_mensagens: list):
    """Executa a busca RAG no ChromaDB e gera resposta em streaming para o diálogo livre com o orientador."""
    llm = assegurar_modelo_valido(llm)
    try:
        docs = vectordb.similarity_search(pergunta, k=3)
        contexto = "\n\n".join([
            f"[Manual: {os.path.basename(doc.metadata.get('source', 'Referência'))}]\n{doc.page_content}"
            for doc in docs
        ]) if docs else ""
    except Exception as e:
        print(f"Aviso na busca vetorial do diálogo: {e}")
        contexto = ""
        
    msgs_recentes = historico_mensagens[-8:] if len(historico_mensagens) > 8 else historico_mensagens
    historico_str = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs_recentes])
    
    prompt = (
        "Você é um orientador e estrategista acadêmico experiente em Ciência da Computação e Metodologia Científica.\n"
        "Sua missão é continuar a conversa com o aluno, tirando dúvidas sobre o projeto de pesquisa, aprofundando o tema e orientando os próximos passos.\n\n"
        f"HISTÓRICO RECENTE DA CONVERSA:\n{historico_str}\n\n"
        f"TRECHOS DE MANUAIS METODOLÓGICOS RELEVANTES:\n{contexto}\n\n"
        f"PERGUNTA DO ALUNO: {pergunta}\n\n"
        "Responda de forma didática, encorajadora, estruturada e em português do Brasil, fundamentando suas recomendações com rigor acadêmico."
    )
    
    try:
        for chunk in llm.stream(prompt):
            txt = extrair_texto_conteudo(chunk)
            if txt:
                yield txt
    except Exception as e:
        if "404" in str(e) or "NOT_FOUND" in str(e) or "2.5" in str(e) or "429" in str(e) or "503" in str(e):
            llm.model = "gemini-3.5-flash-lite"
            for chunk in llm.stream(prompt):
                txt = extrair_texto_conteudo(chunk)
                if txt:
                    yield txt
        else:
            raise e

def inicializar_agente_de_dialogo(llm, vectordb, historico_mensagens: list):
    """Inicializa o agente para diálogo livre pós-estratégia com RAG."""
    return DialogoAcademicoAgent(llm, vectordb, historico_mensagens)

def stream_resposta_rag_estrito(llm, vectordb, pergunta: str, historico_mensagens: list, documento_aluno: str = None):
    """Executa a busca RAG no ChromaDB e gera resposta em streaming com rigor científico e citação de fontes."""
    llm = assegurar_modelo_valido(llm)
    # Se houver documento anexado pelo pesquisador, enriquece a busca com diretrizes metodológicas
    termo_busca = pergunta
    if documento_aluno and documento_aluno.strip():
        termo_busca = f"{pergunta} metodologia cientifica objetivos pesquisa procedimentos academicos"

    docs = vectordb.similarity_search(termo_busca, k=4)
    
    def formatar_nome_fonte(fonte_str: str) -> str:
        b = os.path.basename(fonte_str or "Manual")
        b = re.sub(r'^\++', '', b)
        b = re.sub(r'\.(txt|pdf)$', '', b, flags=re.IGNORECASE)
        return b.strip()

    if docs:
        contexto = "\n\n".join([
            f"--- TRECHO DA LITERATURA ({formatar_nome_fonte(doc.metadata.get('source', 'Manual'))}) ---\n{doc.page_content}"
            for doc in docs
        ])
    else:
        contexto = "Nenhum trecho relevante foi recuperado da base de dados."

    # Prepara histórico resumido recente
    msgs_recentes = historico_mensagens[-6:] if len(historico_mensagens) > 6 else historico_mensagens
    historico_str = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs_recentes])

    bloco_doc = ""
    if documento_aluno and documento_aluno.strip():
        bloco_doc = (
            "\nDOCUMENTO / TEXTO SUBMETIDO PELO PESQUISADOR:\n"
            "```\n"
            f"{documento_aluno[:4000]}\n"
            "```\n"
            "DIRETRIZ DE ANÁLISE: Avalie o texto submetido pelo pesquisador à luz da literatura metodológica fornecida, "
            "apontando congruências, lacunas e recomendações técnicas formais de aprimoramento.\n"
        )

    template_estrito = """Você é um Consultor e Metodólogo Científico sênior, com rigor conceitual e fidelidade à literatura acadêmica.

Sua missão é responder à consulta metodológica do pesquisador com base ESTRITAMENTE nos fragmentos de manuais e livros fornecidos abaixo.
{bloco_documento}
DIRETRIZES DE FIDELIDADE (ANTI-ALUCINAÇÃO):
1. Use SOMENTE os conceitos, normas e métodos presentes nos TRECHOS DA LITERATURA CIENTÍFICA fornecidos. Não invente regras que não constem no texto.
2. Se a informação não constar nos trechos fornecidos, declare: "Com base na literatura acadêmica disponível no sistema, não foi possível encontrar informações suficientes para responder a este ponto específico com exatidão." e recomende qual tipo de literatura buscar.
3. Mantenha tom formal, didático, objetivo e estruturado em Língua Portuguesa (Brasil).
4. REGRA DE FINALIZAÇÃO: NÃO escreva seção de 'Obras e Manuais Consultados' nem lista de referências ao final da resposta; encerre após as recomendações, pois o sistema adiciona a listagem oficial de fontes automaticamente.

TRECHOS DA LITERATURA CIENTÍFICA:
{contexto}

HISTÓRICO DA CONVERSA:
{historico}

CONSULTA DO PESQUISADOR:
{pergunta}

PARECER METODOLÓGICO FUNDAMENTADO:"""

    prompt = PromptTemplate.from_template(template_estrito)
    chain = prompt | llm

    try:
        for chunk in chain.stream({
            "contexto": contexto,
            "historico": historico_str,
            "pergunta": pergunta,
            "bloco_documento": bloco_doc
        }):
            txt = extrair_texto_conteudo(chunk)
            if txt:
                yield txt
    except Exception as e:
        if "404" in str(e) or "NOT_FOUND" in str(e) or "2.5" in str(e) or "429" in str(e) or "503" in str(e):
            llm.model = "gemini-3.5-flash-lite"
            for chunk in (prompt | llm).stream({
                "contexto": contexto,
                "historico": historico_str,
                "pergunta": pergunta,
                "bloco_documento": bloco_doc
            }):
                txt = extrair_texto_conteudo(chunk)
                if txt:
                    yield txt
        else:
            raise e
            
    # Adiciona a citação formatada das fontes encontradas (única listagem oficial)
    fontes_unicas = sorted(list(set(
        formatar_nome_fonte(doc.metadata.get('source', 'Manual'))
        for doc in docs if doc.metadata.get('source')
    )))
    if fontes_unicas:
        fontes_md = "\n\n---\n**Obras e Manuais Consultados na Base:**\n" + "\n".join([f"- *{f}*" for f in fontes_unicas])
        yield fontes_md

