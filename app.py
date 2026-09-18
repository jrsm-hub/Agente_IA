import sys
import os

# Garante que o diretório raiz do projeto esteja no path (essencial para Linux / Streamlit Cloud)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

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
    carregar_llm,
    carregar_llm_estrito,
    gerar_titulo_conversa,
    stream_gerar_proxima_pergunta,
    stream_gerar_documento_estrategico,
    stream_resposta_rag_estrito,
    inicializar_agente_de_dialogo,
    separar_pergunta_e_opcoes
)
from services.pdf_service import criar_pdf_formatado
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Configurações iniciais da aplicação
nest_asyncio.apply()
st.set_page_config(
    page_title="Orientador & Estrategista Acadêmico",
    page_icon="🧭",
    layout="wide"
)

# Inicialização de Recursos Globais
db = init_firebase()
llm = carregar_llm()
llm_estrito = carregar_llm_estrito()
vectordb = carregar_vectordb()

def inicializar_sessao():
    """Inicializa ou reseta o estado da sessão de chat do Streamlit."""
    st.session_state.fase = "SELECAO_MODO"
    st.session_state.tipo_conversa = None
    st.session_state.historico_mensagens = []
    st.session_state.agent_executor = None
    st.session_state.documento_gerado = None
    st.session_state.conversa_id = None

# --- TELA DE LOGIN / IDENTIFICAÇÃO DO USUÁRIO ---
if "user_id" not in st.session_state or not st.session_state.user_id:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_l, col_center, col_r = st.columns([1, 2, 1])
    
    with col_center:
        st.markdown(
            """
            <div style="text-align: center; padding: 20px; border-radius: 12px; background-color: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);">
                <h1 style="margin-bottom: 0px;">🧭 Orientador Acadêmico</h1>
                <p style="color: #888; font-size: 15px; margin-top: 5px;">Ambiente Inteligente de Pesquisa & Metodologia Científica</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")
        st.markdown("#### 🔑 Identificação do Pesquisador / Aluno")
        st.info("Informe seu e-mail institucional ou matrícula para acessar o seu ambiente com histórico protegido.")
        
        identificador = st.text_input(
            "E-mail acadêmico ou Matrícula:",
            placeholder="ex: aluno@universidade.br ou 20261001",
            key="login_user_input"
        )
        
        if st.button("Acessar Meu Ambiente 🚀", use_container_width=True):
            if identificador and identificador.strip():
                st.session_state.user_id = identificador.strip().lower()
                inicializar_sessao()
                st.rerun()
            else:
                st.warning("Por favor, digite seu e-mail ou matrícula para continuar.")
                
    st.stop()

# Garante estado inicial da sessão se logado
if "fase" not in st.session_state:
    inicializar_sessao()

# --- BARRA LATERAL (GESTÃO DO USUÁRIO E CONVERSAS) ---
with st.sidebar:
    st.markdown(f"👤 **Pesquisador:** `{st.session_state.user_id}`")
    
    col_out, col_new = st.columns([1, 1])
    with col_out:
        if st.button("🚪 Sair", use_container_width=True, help="Encerrar sessão ou trocar de usuário"):
            st.session_state.user_id = None
            inicializar_sessao()
            st.rerun()
    with col_new:
        if st.button("➕ Nova", use_container_width=True, help="Criar nova conversa"):
            inicializar_sessao()
            st.rerun()

    st.divider()
    st.subheader("Minhas Conversas")

    conversas = listar_conversas(db, user_id=st.session_state.user_id)
    if not conversas:
        st.caption("Nenhuma conversa salva ainda. Inicie uma nova conversa!")
        
    for conversa in conversas:
        conv_data = conversa.to_dict()
        titulo = conv_data.get('title', 'Conversa')
        conv_id = conversa.id
        eh_ativa = (st.session_state.get("conversa_id") == conv_id)
        label = f"📍 {titulo}" if eh_ativa else titulo
        
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
                else:
                    st.session_state.fase = "COLETA"
            st.rerun()

    # Opção de renomear se houver conversa ativa
    if st.session_state.get("conversa_id"):
        st.divider()
        with st.expander("⚙️ Renomear Conversa Atual"):
            novo_nome = st.text_input("Novo título:", placeholder="Digite o novo título...", key="input_rename")
            if st.button("Salvar Título", use_container_width=True):
                if novo_nome.strip():
                    prefixo = "🧭" if st.session_state.get("tipo_conversa") == "estrategia" else "📚"
                    renomear_conversa(db, st.session_state.conversa_id, f"{prefixo} {novo_nome.strip()}")
                    st.rerun()

# --- TELA DE SELEÇÃO DE MODO (QUANDO NÃO HÁ CONVERSA SELECIONADA) ---
if st.session_state.fase in ["INICIO", "SELECAO_MODO"] and not st.session_state.get("conversa_id"):
    st.markdown("## 🧭 Bem-vindo ao seu Orientador Virtual!")
    st.markdown("Escolha como você gostaria de trabalhar hoje:")
    st.write("")
    
    col_modo1, col_modo2 = st.columns(2)
    
    with col_modo1:
        st.markdown(
            """
            ### 🧭 1. Estratégia de Pesquisa (Ideação)
            Ideal para quem está no início do trabalho ou precisa:
            - **Definir e delimitar** o tema de pesquisa com apoio metodológico.
            - **Construir perguntas** e caminhos metodológicos passo a passo.
            - **Receber sugestões dinâmicas** com opções de escolha rápida.
            - **Gerar Documento Estratégico** completo para download em PDF.
            """
        )
        if st.button("Iniciar Estratégia de Pesquisa 🧭", key="btn_modo_estrategia", use_container_width=True):
            inicializar_sessao()
            mensagem_inicial = (
                "Olá! Sou o seu estrategista de pesquisa. Vamos construir juntos a base para o seu trabalho acadêmico. "
                "Para começar, qual é a sua grande área de interesse?"
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
            ### 📚 2. Consultor Metodológico (RAG Fiel)
            Ideal para tirar dúvidas acadêmicas com máximo rigor científico:
            - **Respostas estritamente ancoradas** nos manuais e livros da base.
            - **Zero Alucinações:** Respostas precisas sobre métodos, normas e etapas.
            - **Citação explícita** dos manuais e obras de referência consultadas.
            - **Chat livre e direto** sem etapas de perguntas pré-definidas.
            """
        )
        if st.button("Iniciar Consultoria Metodológica 📚", key="btn_modo_rag", use_container_width=True):
            inicializar_sessao()
            mensagem_inicial = (
                "Olá! Sou o seu Consultor Metodológico. Estou pronto para esclarecer dúvidas sobre metodologia científica, "
                "estrutura de projetos, tipos de pesquisa, amostragem e normas acadêmicas com base estrita na literatura.\n\n"
                "Qual é a sua dúvida metodológica hoje?"
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
    st.title("🧭 Estrategista de Pesquisa Acadêmica")

    total_msgs = len(st.session_state.get("historico_mensagens", []))

    # Exibição do histórico de mensagens
    for idx, msg in enumerate(st.session_state.get("historico_mensagens", [])):
        with st.chat_message(msg["role"]):
            content = msg.get("content", "")
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
                label="📥 Descarregar Estratégia em PDF",
                data=pdf_bytes,
                file_name="estrategia_de_pesquisa.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with col_info:
            st.caption("✨ *Estratégia completa consolidada e pronta para exportação.*")

    # --- FASE 1.1: COLETA DINÂMICA ---
    if st.session_state.fase == "COLETA":
        if len(st.session_state.historico_mensagens) > 1:
            if st.button("🚀 Já temos informação suficiente. Gerar Documento Estratégico!", use_container_width=True):
                st.session_state.fase = "GERACAO"
                st.rerun()

        resposta_selecionada = None
        if st.session_state.historico_mensagens:
            ultima_msg = st.session_state.historico_mensagens[-1]
            if ultima_msg.get("role") == "assistant":
                _, opcoes = separar_pergunta_e_opcoes(ultima_msg.get("content", ""))
                if opcoes:
                    st.markdown("##### 💡 Escolha uma direção abaixo:")
                    col1, col2 = st.columns(2)
                    for i, opt in enumerate(opcoes):
                        col = col1 if i % 2 == 0 else col2
                        msg_count = len(st.session_state.historico_mensagens)
                        if col.button(f"{i+1}. {opt}", key=f"quick_opt_{i}_{msg_count}", use_container_width=True):
                            resposta_selecionada = f"{i+1}. {opt}"

        prompt_usuario = st.chat_input("Ou digite uma resposta personalizada...")
        entrada_final = resposta_selecionada or prompt_usuario

        if entrada_final:
            with st.chat_message("user"):
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
            
            historico_texto = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.historico_mensagens])
            
            # Gera título na primeira resposta do usuário
            if len(st.session_state.historico_mensagens) == 2:
                novo_titulo = gerar_titulo_conversa(llm, historico_texto)
                if novo_titulo:
                    atualizar_titulo_conversa(db, st.session_state.conversa_id, novo_titulo, prefixo_icone="🧭")

            with st.chat_message("assistant"):
                stream = stream_gerar_proxima_pergunta(llm, historico_texto, vectordb=vectordb)
                proxima_pergunta = st.write_stream(stream)
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
            
            st.rerun()

    # --- FASE 1.2: GERAÇÃO DO DOCUMENTO ---
    elif st.session_state.fase == "GERACAO":
        if st.session_state.documento_gerado is None:
            historico_texto = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.historico_mensagens])
            with st.chat_message("assistant"):
                st.write("✍️ **A redigir a estratégia de pesquisa em tempo real...**")
                stream = stream_gerar_documento_estrategico(llm, historico_texto)
                doc_final = st.write_stream(stream)
                st.session_state.documento_gerado = doc_final
            
            st.session_state.historico_mensagens.append({"role": "assistant", "content": doc_final})
            adicionar_mensagem(db, st.session_state.conversa_id, role="assistant", content=doc_final, fase="GERACAO")
            salvar_documento_estrategico_conversa(db, st.session_state.conversa_id, doc_final)
            
            msg_transicao = (
                "✨ **A sua estratégia de pesquisa foi consolidada acima!**\n\n"
                "Agora estou pronto para aprofundar qualquer ponto, tirar dúvidas sobre metodologia ou sugerir referências. "
                "O que gostaria de explorar primeiro?"
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
        with st.expander("⭐ Avaliar esta Estratégia de Pesquisa"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Útil", use_container_width=True, key="fb_bom_est"):
                    st.session_state.temp_feedback = "bom"
            with col2:
                if st.button("👎 Precisa melhorar", use_container_width=True, key="fb_ruim_est"):
                    st.session_state.temp_feedback = "ruim"

            if "temp_feedback" in st.session_state:
                comentario = st.text_area("O que podemos melhorar?", placeholder="Ex: Mais referências, tema mais específico...")
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

        if prompt_usuario := st.chat_input("Faça uma pergunta sobre a sua estratégia..."):
            with st.chat_message("user"):
                st.markdown(prompt_usuario)
            st.session_state.historico_mensagens.append({"role": "user", "content": prompt_usuario})
            adicionar_mensagem(db, st.session_state.conversa_id, role="user", content=prompt_usuario, fase="DIALOGO_ABERTO")
            
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                try:
                    if not st.session_state.get('agent_executor'):
                        st.session_state.agent_executor = inicializar_agente_de_dialogo(
                            llm, vectordb, st.session_state.historico_mensagens
                        )
                    
                    response = st.session_state.agent_executor.invoke(
                        {"input": prompt_usuario},
                        {"callbacks": [st_callback]}
                    )
                    resposta = response["output"]
                    st.markdown(resposta)
                except Exception as e:
                    resposta = f"Desculpe, ocorreu um erro: {e}"
                    st.error(resposta)
            
            st.session_state.historico_mensagens.append({"role": "assistant", "content": resposta})
            adicionar_mensagem(db, st.session_state.conversa_id, role="assistant", content=resposta, fase="DIALOGO_ABERTO")


# ==============================================================================
# MODO 2: CONSULTOR METODOLÓGICO (RAG ESTRITO & CITAÇÃO DE FONTES)
# ==============================================================================
elif st.session_state.tipo_conversa == "rag_estrito":
    st.title("📚 Consultor Metodológico (RAG Fiel)")
    st.caption("🔬 Respostas fundamentadas com rigor científico e citação de fontes dos manuais acadêmicos.")

    # Exibição do histórico de mensagens
    for msg in st.session_state.get("historico_mensagens", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content", ""))

    # Avaliação do atendimento
    with st.expander("⭐ Avaliar esta Consulta Metodológica"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Resposta Clara e Precisa", use_container_width=True, key="fb_bom_rag"):
                st.session_state.temp_feedback_rag = "bom"
        with col2:
            if st.button("👎 Pouco Precisa / Incompleta", use_container_width=True, key="fb_ruim_rag"):
                st.session_state.temp_feedback_rag = "ruim"

        if "temp_feedback_rag" in st.session_state:
            comentario_rag = st.text_area("O que podemos melhorar?", placeholder="Ex: Citar mais autores, aprofundar exemplos...")
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
    if prompt_usuario := st.chat_input("Faça sua pergunta sobre metodologia científica ou normas..."):
        with st.chat_message("user"):
            st.markdown(prompt_usuario)
        st.session_state.historico_mensagens.append({"role": "user", "content": prompt_usuario})
        adicionar_mensagem(db, st.session_state.conversa_id, role="user", content=prompt_usuario, fase="RAG_ESTRITO")
        
        # Gera título inteligente com base na pergunta
        if len(st.session_state.historico_mensagens) == 2:
            novo_titulo = gerar_titulo_conversa(llm, f"Pergunta do aluno: {prompt_usuario}")
            if novo_titulo:
                atualizar_titulo_conversa(db, st.session_state.conversa_id, novo_titulo, prefixo_icone="📚")

        # Stream da resposta RAG com citação de fontes
        with st.chat_message("assistant"):
            stream = stream_resposta_rag_estrito(
                llm_estrito, vectordb, prompt_usuario, st.session_state.historico_mensagens
            )
            resposta_completa = st.write_stream(stream)
            st.session_state.historico_mensagens.append({"role": "assistant", "content": resposta_completa})
            adicionar_mensagem(db, st.session_state.conversa_id, role="assistant", content=resposta_completa, fase="RAG_ESTRITO")
        
        st.rerun()