import sys
import os

# Garante que o diretório raiz do projeto esteja no path (essencial para Linux / Streamlit Cloud)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pymupdf as fitz
import nest_asyncio
import streamlit as st
import streamlit.components.v1 as components

from services.firebase_service import (
    init_firebase,
    criar_conversa,
    listar_conversas,
    carregar_mensagens_conversa,
    obter_dados_conversa,
    adicionar_mensagem,
    salvar_documento_estrategico_conversa,
    atualizar_titulo_conversa,
    renomear_conversa,
    salvar_feedback
)
from services.rag_service import carregar_vectordb
from services.ai_service import (
    assegurar_modelo_valido,
    carregar_llm,
    carregar_llm_estrito,
    gerar_titulo_conversa,
    extrair_tema_documento_estrategico,
    gerar_proxima_pergunta_dinamica,
    stream_gerar_proxima_pergunta,
    stream_gerar_documento_estrategico,
    stream_resposta_rag_estrito,
    stream_dialogo_academico,
    inicializar_agente_de_dialogo,
    separar_pergunta_e_opcoes,
    extrair_texto_conteudo
)
from services.pdf_service import criar_pdf_formatado

# Configurações iniciais da aplicação
nest_asyncio.apply()
st.set_page_config(
    page_title="Orientador Acadêmico | Mediação Científica",
    page_icon=None,
    layout="wide"
)

# Estilização para garantir que os botões de seleção exibam o texto completo sem truncar
st.markdown("""
<style>
/* Permite quebra de linha e visualização integral do texto nos botões de opções */
div[data-testid="stButton"] button,
button[data-testid^="baseButton-"],
button[data-testid^="stBaseButton-"],
button[kind="secondary"],
.stButton > button {
    height: auto !important;
    min-height: 52px !important;
    max-height: none !important;
    padding: 12px 18px !important;
    white-space: normal !important;
    word-break: normal !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    text-align: left !important;
    display: block !important;
    border-radius: 8px !important;
    line-height: 1.5 !important;
    overflow: visible !important;
}

div[data-testid="stButton"] button *,
button[data-testid^="baseButton-"] *,
button[data-testid^="stBaseButton-"] *,
button[kind="secondary"] *,
.stButton > button * {
    white-space: normal !important;
    word-break: normal !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    text-overflow: unset !important;
    overflow: visible !important;
    text-align: left !important;
    line-height: 1.5 !important;
}
</style>
""", unsafe_allow_html=True)

from config import GOOGLE_API_KEY, LLM_MODEL

# Inicialização de Recursos Globais com invalidação automática de cache
db = init_firebase()
llm = carregar_llm(api_key=GOOGLE_API_KEY, model_name=LLM_MODEL)
llm_estrito = carregar_llm_estrito(api_key=GOOGLE_API_KEY, model_name=LLM_MODEL)
llm = assegurar_modelo_valido(llm)
llm_estrito = assegurar_modelo_valido(llm_estrito)
vectordb = carregar_vectordb(api_key=GOOGLE_API_KEY)

# Identidade visual sóbria para o chat acadêmico
AVATAR_ORIENTADOR = "🎓"
AVATAR_ALUNO = "👤"

def extrair_texto_documento(arquivo) -> str:
    """Extrai texto de arquivo PDF ou TXT carregado via Streamlit com getvalue seguro."""
    try:
        nome = arquivo.name.lower()
        bytes_data = arquivo.getvalue() if hasattr(arquivo, "getvalue") else arquivo.read()
        if not bytes_data:
            return ""
        if nome.endswith(".pdf"):
            doc = fitz.open(stream=bytes_data, filetype="pdf")
            texto_paginas = []
            for pagina in doc:
                txt = pagina.get_text()
                if txt.strip():
                    texto_paginas.append(txt)
            texto_completo = "\n".join(texto_paginas).strip()
            if not texto_completo:
                st.warning("O arquivo PDF foi carregado, mas não contém texto selecionável (pode ser uma imagem escaneada). Por favor, utilize um PDF com camada de texto ou arquivo .txt.")
            return texto_completo
        elif nome.endswith(".txt"):
            return bytes_data.decode("utf-8", errors="replace").strip()
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
    return ""

def exibir_erro_amigavel(e: Exception):
    """Exibe mensagem clara e orientações passo a passo em caso de falha de API ou permissão."""
    msg = str(e)
    nome = type(e).__name__
    
    chave_ativa = GOOGLE_API_KEY or ""
    if chave_ativa and len(chave_ativa) >= 8:
        chave_mascarada = f"`{chave_ativa[:4]}...{chave_ativa[-4:]}` (total {len(chave_ativa)} chars)"
    else:
        chave_mascarada = "`Não encontrada ou vazia`"

    if "PermissionDenied" in nome or "PermissionDenied" in msg or "403" in msg:
        st.error(
            f"**Erro de Permissão na API do Google (403 Permission Denied):**\n\n"
            f"A chave ativa carregada no momento é: **{chave_mascarada}**.\n\n"
            "**Verificações no Streamlit Cloud:**\n"
            "1. **A chave ativa acima é a sua chave nova?**\n"
            "   - Se ela for a chave antiga (ou se estiver vazia), o Streamlit Cloud ainda não carregou a nova chave da nuvem.\n"
            "   - No Streamlit Cloud, clique no canto inferior direito em **Manage app** (ou nos 3 pontinhos) > **Settings** > **Secrets**.\n"
            "   - Certifique-se de que `GOOGLE_API_KEY = \"AIzaSy...\"` esteja digitada corretamente.\n"
            "   - Clique em **Save** e em seguida selecione obrigatoriamente **Reboot App** no menu inferior direito para limpar a memória da nuvem.\n"
            "2. **A chave nova foi gerada no Google AI Studio?**\n"
            "   - Certifique-se de que a chave foi criada no [Google AI Studio](https://aistudio.google.com/) e que a conta não possui restrições de faturamento ou de idade."
        )
    else:
        st.error(f"Erro na comunicação com a API: {e}")

def inicializar_sessao():
    """Inicializa ou reseta o estado da sessão de atendimento do Streamlit."""
    st.session_state.fase = "SELECAO_MODO"
    st.session_state.tipo_conversa = None
    st.session_state.historico_mensagens = []
    st.session_state.agent_executor = None
    st.session_state.documento_gerado = None
    st.session_state.conversa_id = None
    st.session_state.doc_anexado_texto = None
    st.session_state.doc_anexado_nome = None

# --- TELA DE LOGIN / IDENTIFICAÇÃO DO USUÁRIO ---
if "user_id" not in st.session_state or not st.session_state.user_id:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_l, col_center, col_r = st.columns([1, 2, 1])
    
    with col_center:
        st.markdown(
            """
            <div style="text-align: center; padding: 24px; border-radius: 8px; background-color: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);">
                <h2 style="margin-bottom: 4px; font-weight: 600; letter-spacing: -0.5px;">Orientador Acadêmico</h2>
                <p style="color: #94a3b8; font-size: 14px; margin-top: 0px;">Ambiente de Mediação Pedagógica & Estruturação Metodológica</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")
        st.markdown("##### Identificação do Pesquisador")
        st.caption("Informe seu e-mail institucional ou matrícula para acessar o seu ambiente com histórico protegido.")
        
        identificador = st.text_input(
            "E-mail institucional ou Matrícula:",
            placeholder="ex: aluno@universidade.br ou 20261001",
            key="login_user_input"
        )
        
        if st.button("Acessar Sistema", use_container_width=True):
            if identificador and identificador.strip():
                st.session_state.user_id = identificador.strip().lower()
                inicializar_sessao()
                st.rerun()
            else:
                st.warning("Informe sua identificação para continuar.")
                
    st.stop()

# Garante estado inicial da sessão se logado
if "fase" not in st.session_state:
    inicializar_sessao()

# --- BARRA LATERAL (GESTÃO DO USUÁRIO E ATENDIMENTOS) ---
with st.sidebar:
    st.markdown(f"**Pesquisador:** `{st.session_state.user_id}`")
    
    col_out, col_new = st.columns([1, 1])
    with col_out:
        if st.button("Sair", use_container_width=True, help="Encerrar sessão atual"):
            st.session_state.user_id = None
            inicializar_sessao()
            st.rerun()
    with col_new:
        if st.button("Nova Conversa", use_container_width=True, help="Iniciar nova sessão"):
            inicializar_sessao()
            st.rerun()

    st.divider()
    st.subheader("Atendimentos Salvos")

    conversas = listar_conversas(db, user_id=st.session_state.user_id)
    if not conversas:
        st.caption("Nenhum atendimento registrado. Inicie uma nova conversa.")
        
    for conversa in conversas:
        conv_data = conversa.to_dict()
        titulo = conv_data.get('title', 'Conversa')
        
        # Recuperação retroativa: se a conversa já possui documento estratégico mas o título ainda é genérico
        if ("Atendimento " in titulo or titulo == "Conversa") and conv_data.get('documento_gerado'):
            tema_recup = extrair_tema_documento_estrategico(conv_data.get('documento_gerado'))
            if tema_recup:
                titulo = f"[Estratégia] {tema_recup}"
                atualizar_titulo_conversa(db, conversa.id, tema_recup, prefixo_icone="[Estratégia]")

        # Remove emojis legados salvos no banco para manter a sobriedade
        for em in ["🧭", "📚", "📍", "🚀", "💡", "⭐", "✨"]:
            titulo = titulo.replace(em, "").strip()
        conv_id = conversa.id
        eh_ativa = (st.session_state.get("conversa_id") == conv_id)
        label = f"[Ativa] {titulo}" if eh_ativa else titulo
        
        if st.button(label, key=f"btn_{conv_id}", use_container_width=True):
            inicializar_sessao()
            st.session_state.conversa_id = conv_id
            st.session_state.historico_mensagens = carregar_mensagens_conversa(db, conv_id)
            
            tipo_salvo = conv_data.get('tipo', 'estrategia')
            st.session_state.tipo_conversa = tipo_salvo
            
            if tipo_salvo == "rag_estrito":
                st.session_state.fase = "RAG_ESTRITO"
            else:
                # Detecta se já existe documento estratégico gravado no Firestore ou no histórico
                doc_salvo = conv_data.get('documento_gerado')
                if not doc_salvo:
                    doc_salvo = next(
                        (msg['content'] for msg in st.session_state.historico_mensagens if "Estratégia de Pesquisa" in msg.get('content', '')),
                        None
                    )
                if doc_salvo:
                    st.session_state.fase = "DIALOGO_ABERTO"
                    st.session_state.documento_gerado = doc_salvo
                    if "Atendimento" in titulo or titulo == "Conversa":
                        tema_recup = extrair_tema_documento_estrategico(doc_salvo)
                        if tema_recup:
                            atualizar_titulo_conversa(db, conv_id, tema_recup, prefixo_icone="[Estratégia]")
                else:
                    st.session_state.fase = "COLETA"
            st.rerun()

    # Opção de renomear ou detectar tema se houver conversa ativa
    if st.session_state.get("conversa_id"):
        st.divider()
        with st.expander("Gerenciar Título do Atendimento"):
            if st.button("Identificar e Salvar Tema de Pesquisa", use_container_width=True, key="btn_auto_detect_tema"):
                historico_texto_atual = "\n".join([f"{m['role']}: {extrair_texto_conteudo(m.get('content', ''))}" for m in st.session_state.historico_mensagens])
                doc = st.session_state.get("documento_gerado")
                tema_auto = extrair_tema_documento_estrategico(doc) if doc else None
                if not tema_auto:
                    tema_auto = gerar_titulo_conversa(llm, historico_texto_atual)
                if tema_auto:
                    prefixo = "[Estratégia]" if st.session_state.get("tipo_conversa") == "estrategia" else "[Metodologia]"
                    atualizar_titulo_conversa(db, st.session_state.conversa_id, tema_auto, prefixo_icone=prefixo)
                    st.success(f"Título salvo: {tema_auto}")
                    st.rerun()

            novo_nome = st.text_input("Ou renomear manualmente:", placeholder="Digite o novo título...", key="input_rename")
            if st.button("Salvar Título Manual", use_container_width=True, key="btn_salvar_manual"):
                if novo_nome.strip():
                    prefixo = "[Estratégia]" if st.session_state.get("tipo_conversa") == "estrategia" else "[Metodologia]"
                    renomear_conversa(db, st.session_state.conversa_id, f"{prefixo} {novo_nome.strip()}")
                    st.rerun()

# --- TELA DE SELEÇÃO DE MODO (QUANDO NÃO HÁ CONVERSA SELECIONADA) ---
if st.session_state.fase in ["INICIO", "SELECAO_MODO"] and not st.session_state.get("conversa_id"):
    st.markdown("## Orientador Acadêmico")
    st.markdown("Selecione o módulo de trabalho:")
    st.write("")
    
    col_modo1, col_modo2 = st.columns(2)
    
    with col_modo1:
        st.markdown(
            """
            ### 1. Estratégia de Pesquisa (Ideação e Afunilamento)
            Condução dialógica e estruturação de projetos:
            - **Definição e delimitação** do tema de pesquisa com fundamentação metodológica.
            - **Formulação socrática** de perguntas, objetivos e hipóteses passo a passo.
            - **Aprofundamento guiado** com alternativas e direções metodológicas.
            - **Consolidação em Documento Estratégico** formal para exportação em PDF.
            """
        )
        if st.button("Iniciar Formulação Estratégica", key="btn_modo_estrategia", use_container_width=True):
            inicializar_sessao()
            mensagem_inicial = (
                "Seja bem-vindo. Sou o seu orientador para estruturação metodológica. "
                "Conduziremos um diálogo estruturado para delimitar seu tema e formalizar uma proposta de pesquisa exequível. "
                "Para iniciarmos, qual é a sua área geral de interesse ou problema preliminar?"
            )
            conversa_id = criar_conversa(db, st.session_state.user_id, mensagem_inicial, tipo="estrategia")
            st.session_state.conversa_id = conversa_id
            st.session_state.tipo_conversa = "estrategia"
            st.session_state.fase = "COLETA"
            st.session_state.historico_mensagens = [{"role": "assistant", "content": mensagem_inicial}]
            st.rerun()

    with col_modo2:
        st.markdown(
            """
            ### 2. Consultoria Metodológica & Auditoria Textual (RAG)
            Consulta rigorosa à literatura científica e avaliação de rascunhos:
            - **Respostas ancoradas estritamente** em manuais canônicos de pesquisa científica.
            - **Mitigação de alucinações:** Esclarecimento de normas, métodos, amostragem e procedimentos.
            - **Leitor de Documentos:** Submissão de rascunhos (PDF/TXT) para parecer metodológico fundamentado.
            - **Citação formal** das obras e autores consultados na base.
            """
        )
        if st.button("Acessar Consultoria Metodológica", key="btn_modo_rag", use_container_width=True):
            inicializar_sessao()
            mensagem_inicial = (
                "Ambiente de Consultoria Metodológica ativo. Estou fundamentado estritamente na literatura canônica "
                "de metodologia científica para esclarecer dúvidas sobre delineamento de pesquisa, normas acadêmicas e procedimentos técnicos. "
                "Você também pode submeter um documento (PDF ou TXT) para auditoria metodológica.\n\n"
                "Qual é a sua consulta ou necessidade metodológica?"
            )
            conversa_id = criar_conversa(db, st.session_state.user_id, mensagem_inicial, tipo="rag_estrito")
            st.session_state.conversa_id = conversa_id
            st.session_state.tipo_conversa = "rag_estrito"
            st.session_state.fase = "RAG_ESTRITO"
            st.session_state.historico_mensagens = [{"role": "assistant", "content": mensagem_inicial}]
            st.rerun()

    st.stop()


# ==============================================================================
# MODO 1: ESTRATEGISTA DE PESQUISA (COLETA, GERAÇÃO E DIÁLOGO)
# ==============================================================================
if st.session_state.tipo_conversa == "estrategia":
    st.title("Estratégia de Pesquisa Acadêmica")

    total_msgs = len(st.session_state.get("historico_mensagens", []))

    # Exibição do histórico de mensagens
    for idx, msg in enumerate(st.session_state.get("historico_mensagens", [])):
        avatar_atual = AVATAR_ORIENTADOR if msg.get("role") == "assistant" else AVATAR_ALUNO
        with st.chat_message(msg["role"], avatar=avatar_atual):
            content = extrair_texto_conteudo(msg.get("content", ""))
            if msg.get("role") == "assistant":
                # Oculta opções do balão apenas na última mensagem da coleta ativa (onde os botões estão logo abaixo)
                eh_ultima_coleta = (idx == total_msgs - 1) and (st.session_state.fase == "COLETA")
                if eh_ultima_coleta:
                    pergunta_limpa, opcoes = separar_pergunta_e_opcoes(content)
                    st.markdown(pergunta_limpa if (opcoes and pergunta_limpa) else content)
                else:
                    st.markdown(content)
            else:
                st.markdown(content)

    # Destaque permanente do Documento Estratégico com Download em PDF
    if st.session_state.documento_gerado:
        st.divider()
        col_dl, col_info = st.columns([1, 2])
        with col_dl:
            pdf_bytes = criar_pdf_formatado(st.session_state.documento_gerado)
            st.download_button(
                label="Baixar Estratégia em PDF",
                data=pdf_bytes,
                file_name="estrategia_de_pesquisa.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with col_info:
            st.caption("*Documento estratégico compilado e pronto para exportação.*")

    # --- FASE 1.1: COLETA DINÂMICA ---
    if st.session_state.fase == "COLETA":
        if len(st.session_state.historico_mensagens) > 1:
            if st.button("Concluir Coleta e Gerar Documento Estratégico", use_container_width=True):
                st.session_state.fase = "GERACAO"
                st.rerun()

        resposta_selecionada = None
        if st.session_state.historico_mensagens:
            ultima_msg = st.session_state.historico_mensagens[-1]
            if ultima_msg.get("role") == "assistant":
                _, opcoes = separar_pergunta_e_opcoes(ultima_msg.get("content", ""))
                if opcoes:
                    st.markdown("##### Direções metodológicas recomendadas (clique para responder):")
                    for i, opt in enumerate(opcoes):
                        msg_count = len(st.session_state.historico_mensagens)
                        rotulo = f"**{i+1}.** {opt}"
                        if st.button(rotulo, key=f"quick_opt_{i}_{msg_count}", use_container_width=True):
                            resposta_selecionada = f"{i+1}. {opt}"

        prompt_usuario = st.chat_input("Ou digite uma resposta personalizada...")
        entrada_final = resposta_selecionada or prompt_usuario

        if entrada_final:
            with st.chat_message("user", avatar=AVATAR_ALUNO):
                st.markdown(entrada_final)
            st.session_state.historico_mensagens.append({"role": "user", "content": entrada_final})
            
            rodada_atual = (len(st.session_state.historico_mensagens) // 2)
            adicionar_mensagem(
                db,
                st.session_state.conversa_id,
                role="user",
                content=entrada_final,
                fase="COLETA",
                rodada=rodada_atual,
                opcao_selecionada=resposta_selecionada
            )
            
            historico_texto = "\n".join([f"{m['role']}: {extrair_texto_conteudo(m.get('content', ''))}" for m in st.session_state.historico_mensagens])

            with st.chat_message("assistant", avatar=AVATAR_ORIENTADOR):
                with st.spinner("Formulando a próxima pergunta metodológica e opções..."):
                    try:
                        proxima_pergunta = gerar_proxima_pergunta_dinamica(llm, historico_texto)
                        st.session_state.historico_mensagens.append({"role": "assistant", "content": proxima_pergunta})
                        
                        _, opcoes_novas = separar_pergunta_e_opcoes(proxima_pergunta)
                        adicionar_mensagem(
                            db,
                            st.session_state.conversa_id,
                            role="assistant",
                            content=proxima_pergunta,
                            fase="COLETA",
                            rodada=rodada_atual + 1,
                            opcoes_oferecidas=opcoes_novas
                        )
                    except Exception as e:
                        exibir_erro_amigavel(e)
                        st.stop()

            # Atualiza o título na barra lateral com o tema de pesquisa delimitado
            conv_data_atual = obter_dados_conversa(db, st.session_state.conversa_id)
            titulo_atual = conv_data_atual.get("title", "")
            if "Atendimento" in titulo_atual or len(st.session_state.historico_mensagens) <= 5:
                novo_titulo = gerar_titulo_conversa(llm, historico_texto)
                if novo_titulo and "Atendimento" not in novo_titulo:
                    atualizar_titulo_conversa(db, st.session_state.conversa_id, novo_titulo, prefixo_icone="[Estratégia]")
            
            st.rerun()

    # --- FASE 1.2: GERAÇÃO DO DOCUMENTO ---
    elif st.session_state.fase == "GERACAO":
        if st.session_state.documento_gerado is None:
            historico_texto = "\n".join([f"{m['role']}: {extrair_texto_conteudo(m.get('content', ''))}" for m in st.session_state.historico_mensagens])
            with st.chat_message("assistant"):
                st.write("**Redigindo a estratégia de pesquisa em tempo real...**")
                try:
                    stream = stream_gerar_documento_estrategico(llm, historico_texto)
                    doc_final = st.write_stream(stream)
                    st.session_state.documento_gerado = doc_final
                except Exception as e:
                    exibir_erro_amigavel(e)
                    st.stop()
            
            st.session_state.historico_mensagens.append({"role": "assistant", "content": doc_final})
            adicionar_mensagem(db, st.session_state.conversa_id, role="assistant", content=doc_final, fase="GERACAO")
            salvar_documento_estrategico_conversa(db, st.session_state.conversa_id, doc_final)
            
            # Atualiza o título da barra lateral com o tema de pesquisa consolidado no Documento Estratégico
            tema_final = extrair_tema_documento_estrategico(doc_final)
            if not tema_final:
                tema_final = gerar_titulo_conversa(llm, doc_final[:1500])
            if tema_final:
                atualizar_titulo_conversa(db, st.session_state.conversa_id, tema_final, prefixo_icone="[Estratégia]")
            
            msg_transicao = (
                "**Estratégia de pesquisa consolidada com sucesso.**\n\n"
                "O canal de diálogo permanece ativo para aprofundar pontos específicos, esclarecer procedimentos técnicos "
                "ou sanar dúvidas sobre execução e metodologia. O que gostaria de analisar a seguir?"
            )
            st.session_state.historico_mensagens.append({"role": "assistant", "content": msg_transicao})
            adicionar_mensagem(db, st.session_state.conversa_id, role="assistant", content=msg_transicao, fase="DIALOGO_ABERTO")
            
            st.session_state.fase = "DIALOGO_ABERTO"
            st.session_state.agent_executor = inicializar_agente_de_dialogo(
                llm, vectordb, st.session_state.historico_mensagens
            )
            st.rerun()
        else:
            st.session_state.fase = "DIALOGO_ABERTO"
            st.rerun()

    # --- FASE 1.3: DIÁLOGO ABERTO PÓS-ESTRATÉGIA ---
    elif st.session_state.fase == "DIALOGO_ABERTO":
        with st.expander("Avaliação da Estratégia"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Satisfatório", use_container_width=True, key="fb_bom_est"):
                    st.session_state.temp_feedback = "bom"
            with col2:
                if st.button("Necessita Ajustes", use_container_width=True, key="fb_ruim_est"):
                    st.session_state.temp_feedback = "ruim"

            if "temp_feedback" in st.session_state:
                comentario = st.text_area("Observações metodológicas ou sugestões de melhoria:", placeholder="Ex: Detalhar mais as referências, especificar o recorte...")
                if st.button("Submeter Avaliação", key="btn_sub_fb_est"):
                    sucesso, msg_fb = salvar_feedback(
                        db, st.session_state.conversa_id, st.session_state.user_id, st.session_state.temp_feedback, comentario
                    )
                    if sucesso:
                        st.success(msg_fb)
                    else:
                        st.error(msg_fb)
                    del st.session_state.temp_feedback

        st.divider()

        if prompt_usuario := st.chat_input("Faça uma pergunta sobre a sua estratégia de pesquisa..."):
            with st.chat_message("user", avatar=AVATAR_ALUNO):
                st.markdown(prompt_usuario)
            st.session_state.historico_mensagens.append({"role": "user", "content": prompt_usuario})
            adicionar_mensagem(db, st.session_state.conversa_id, role="user", content=prompt_usuario, fase="DIALOGO_ABERTO")
            
            with st.chat_message("assistant", avatar=AVATAR_ORIENTADOR):
                try:
                    stream = stream_dialogo_academico(
                        llm, vectordb, prompt_usuario, st.session_state.historico_mensagens
                    )
                    resposta = st.write_stream(stream)
                except Exception as e:
                    exibir_erro_amigavel(e)
                    st.stop()
            
            st.session_state.historico_mensagens.append({"role": "assistant", "content": resposta})
            adicionar_mensagem(db, st.session_state.conversa_id, role="assistant", content=resposta, fase="DIALOGO_ABERTO")
            st.rerun()


# ==============================================================================
# MODO 2: CONSULTOR METODOLÓGICO & AUDITORIA TEXTUAL (RAG ESTRITO & DOCUMENTOS)
# ==============================================================================
elif st.session_state.tipo_conversa == "rag_estrito":
    st.title("Consultor Metodológico & Auditoria Textual")
    st.caption("Respostas fundamentadas com rigor conceitual na literatura canônica de metodologia científica.")

    # Componente de Leitura de Documentos (PDF ou TXT)
    with st.expander("Submeter Documento para Auditoria Metodológica (PDF ou TXT)", expanded=True):
        col_up1, col_up2 = st.columns([3, 1])
        with col_up1:
            arquivo_submetido = st.file_uploader(
                "Carregue seu anteprojeto, rascunho de metodologia ou introdução para análise:",
                type=["pdf", "txt"],
                key="uploader_doc_rag"
            )
            if arquivo_submetido is not None:
                if st.session_state.get("doc_anexado_nome") != arquivo_submetido.name:
                    texto_extraido = extrair_texto_documento(arquivo_submetido)
                    if texto_extraido:
                        st.session_state.doc_anexado_texto = texto_extraido
                        st.session_state.doc_anexado_nome = arquivo_submetido.name
                        st.rerun()

        with col_up2:
            if st.session_state.get("doc_anexado_nome"):
                st.write(f"**Arquivo ativo:** `{st.session_state.doc_anexado_nome}`")
                st.caption(f"{len(st.session_state.doc_anexado_texto)} caracteres extraídos.")
                if st.button("Remover Documento", use_container_width=True, key="btn_clear_doc"):
                    st.session_state.doc_anexado_texto = None
                    st.session_state.doc_anexado_nome = None
                    st.rerun()
            else:
                st.caption("Nenhum arquivo anexado no momento.")

    if st.session_state.get("doc_anexado_nome"):
        st.info(f"Documento ativo: **{st.session_state.doc_anexado_nome}** ({len(st.session_state.doc_anexado_texto)} caracteres extraídos com sucesso).")
        if st.button("Emitir Parecer Metodológico Completo do Documento", key="btn_auditar_doc_auto", use_container_width=True):
            prompt_auditoria = "Por favor, realize uma auditoria metodológica rigorosa e fundamentada deste documento à luz dos manuais de pesquisa científica."
            st.session_state.historico_mensagens.append({"role": "user", "content": prompt_auditoria})
            adicionar_mensagem(db, st.session_state.conversa_id, role="user", content=prompt_auditoria, fase="RAG_ESTRITO")
            
            with st.chat_message("assistant", avatar=AVATAR_ORIENTADOR):
                try:
                    stream = stream_resposta_rag_estrito(
                        llm_estrito,
                        vectordb,
                        prompt_auditoria,
                        st.session_state.historico_mensagens,
                        documento_aluno=st.session_state.doc_anexado_texto
                    )
                    resposta_completa = st.write_stream(stream)
                    st.session_state.historico_mensagens.append({"role": "assistant", "content": resposta_completa})
                    adicionar_mensagem(db, st.session_state.conversa_id, role="assistant", content=resposta_completa, fase="RAG_ESTRITO")
                except Exception as e:
                    exibir_erro_amigavel(e)
                    st.stop()
            st.rerun()

    # Exibição do histórico de mensagens
    for msg in st.session_state.get("historico_mensagens", []):
        avatar_atual = AVATAR_ORIENTADOR if msg.get("role") == "assistant" else AVATAR_ALUNO
        with st.chat_message(msg["role"], avatar=avatar_atual):
            st.markdown(extrair_texto_conteudo(msg.get("content", "")))

    # Avaliação do atendimento
    with st.expander("Avaliação da Consulta Metodológica"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clara e Fundamentada", use_container_width=True, key="fb_bom_rag"):
                st.session_state.temp_feedback_rag = "bom"
        with col2:
            if st.button("Pouco Precisa ou Incompleta", use_container_width=True, key="fb_ruim_rag"):
                st.session_state.temp_feedback_rag = "ruim"

        if "temp_feedback_rag" in st.session_state:
            comentario_rag = st.text_area("Observações metodológicas ou sugestões:", placeholder="Ex: Citar mais autores, aprofundar exemplos...")
            if st.button("Submeter Avaliação", key="btn_sub_fb_rag"):
                sucesso, msg_fb = salvar_feedback(
                    db, st.session_state.conversa_id, st.session_state.user_id, st.session_state.temp_feedback_rag, comentario_rag
                )
                if sucesso:
                    st.success(msg_fb)
                else:
                    st.error(msg_fb)
                del st.session_state.temp_feedback_rag

    st.divider()

    # Chat direto com RAG Estrito em Streaming
    placeholder_input = "Faça sua consulta metodológica ou solicite parecer sobre o documento..." if st.session_state.get("doc_anexado_nome") else "Faça sua pergunta sobre metodologia científica ou normas..."
    if prompt_usuario := st.chat_input(placeholder_input):
        with st.chat_message("user", avatar=AVATAR_ALUNO):
            st.markdown(prompt_usuario)
        st.session_state.historico_mensagens.append({"role": "user", "content": prompt_usuario})
        adicionar_mensagem(db, st.session_state.conversa_id, role="user", content=prompt_usuario, fase="RAG_ESTRITO")

        # Stream da resposta RAG com citação de fontes e documento
        with st.chat_message("assistant", avatar=AVATAR_ORIENTADOR):
            try:
                stream = stream_resposta_rag_estrito(
                    llm_estrito,
                    vectordb,
                    prompt_usuario,
                    st.session_state.historico_mensagens,
                    documento_aluno=st.session_state.get("doc_anexado_texto")
                )
                resposta_completa = st.write_stream(stream)
                st.session_state.historico_mensagens.append({"role": "assistant", "content": resposta_completa})
                adicionar_mensagem(db, st.session_state.conversa_id, role="assistant", content=resposta_completa, fase="RAG_ESTRITO")
            except Exception as e:
                exibir_erro_amigavel(e)
                st.stop()

        # Atualiza título inteligente com base na consulta se ainda for genérico
        conv_data_atual = obter_dados_conversa(db, st.session_state.conversa_id)
        titulo_atual = conv_data_atual.get("title", "")
        if "Atendimento" in titulo_atual or len(st.session_state.historico_mensagens) <= 3:
            novo_titulo = gerar_titulo_conversa(llm, f"Consulta: {prompt_usuario}")
            if novo_titulo and "Atendimento" not in novo_titulo:
                atualizar_titulo_conversa(db, st.session_state.conversa_id, novo_titulo, prefixo_icone="[Metodologia]")
        
        st.rerun()