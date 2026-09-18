import os
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from config import FIREBASE_CREDENTIALS_PATH

@st.cache_resource
def init_firebase():
    """Inicializa o cliente do Firebase Firestore."""
    try:
        creds_json = st.secrets["firebase_credentials"]
        cred_dict = dict(creds_json)
        cred = credentials.Certificate(cred_dict)
    except (FileNotFoundError, KeyError):
        if not os.path.exists(FIREBASE_CREDENTIALS_PATH):
            st.error(f"Ficheiro de credenciais do Firebase ({FIREBASE_CREDENTIALS_PATH}) não encontrado!")
            st.stop()
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
        
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return firestore.client()

def criar_conversa(db, user_id: str, mensagem_inicial: str, tipo: str = "estrategia", opcoes_iniciais: list = None) -> str:
    """Cria uma nova conversa vinculada a um usuário e insere a mensagem inicial estruturada."""
    nova_conversa_ref = db.collection('conversas').document()
    conversa_id = nova_conversa_ref.id
    
    icone = "🧭" if tipo == "estrategia" else "📚"
    titulo_padrao = f"{icone} Conversa {conversa_id[:4]}..."
    
    nova_conversa_ref.set({
        'user_id': user_id,
        'tipo': tipo,
        'status': 'em_andamento',
        'title': titulo_padrao,
        'total_mensagens': 1,
        'documento_gerado': None,
        'created_at': firestore.SERVER_TIMESTAMP,
        'updated_at': firestore.SERVER_TIMESTAMP
    })
    
    msg_payload = {
        'role': 'assistant',
        'content': mensagem_inicial,
        'fase': 'COLETA' if tipo == 'estrategia' else 'RAG_ESTRITO',
        'rodada': 1,
        'timestamp': firestore.SERVER_TIMESTAMP
    }
    if opcoes_iniciais:
        msg_payload['opcoes_oferecidas'] = opcoes_iniciais
        
    nova_conversa_ref.collection('mensagens').add(msg_payload)
    return conversa_id

def listar_conversas(db, user_id: str = None):
    """Retorna a lista de conversas do usuário ordenadas pela data mais recente."""
    if user_id:
        # Busca filtrada por usuário e ordena em memória para compatibilidade imediata
        docs = db.collection('conversas').where('user_id', '==', user_id).stream()
    else:
        docs = db.collection('conversas').order_by('updated_at', direction=firestore.Query.DESCENDING).stream()
    
    lista = list(docs)
    # Ordena pelo timestamp mais recente (updated_at ou created_at)
    lista.sort(
        key=lambda d: (d.to_dict().get('updated_at') or d.to_dict().get('timestamp') or d.to_dict().get('created_at') or 0),
        reverse=True
    )
    return lista

def carregar_mensagens_conversa(db, conversa_id: str):
    """Carrega todas as mensagens de uma conversa ordenadas por timestamp."""
    msgs_db = db.collection('conversas').document(conversa_id).collection('mensagens').order_by('timestamp').stream()
    return [msg.to_dict() for msg in msgs_db]

def obter_dados_conversa(db, conversa_id: str) -> dict:
    """Retorna os metadados da conversa (título, tipo, status, documento_gerado, user_id)."""
    if not conversa_id:
        return {}
    doc = db.collection('conversas').document(conversa_id).get()
    return doc.to_dict() if doc.exists else {}

def adicionar_mensagem(
    db,
    conversa_id: str,
    role: str,
    content: str,
    fase: str = None,
    rodada: int = None,
    opcoes_oferecidas: list = None,
    opcao_selecionada: str = None,
    fontes_consultadas: list = None
):
    """Salva uma nova mensagem estruturada no Firestore e atualiza updated_at e total_mensagens."""
    if not conversa_id:
        return
    
    msg_payload = {
        'role': role,
        'content': content,
        'timestamp': firestore.SERVER_TIMESTAMP
    }
    if fase:
        msg_payload['fase'] = fase
    if rodada is not None:
        msg_payload['rodada'] = rodada
    if opcoes_oferecidas:
        msg_payload['opcoes_oferecidas'] = opcoes_oferecidas
    if opcao_selecionada:
        msg_payload['opcao_selecionada'] = opcao_selecionada
    if fontes_consultadas:
        msg_payload['fontes_consultadas'] = fontes_consultadas
        
    db.collection('conversas').document(conversa_id).collection('mensagens').add(msg_payload)
    
    try:
        db.collection('conversas').document(conversa_id).update({
            'updated_at': firestore.SERVER_TIMESTAMP,
            'total_mensagens': firestore.Increment(1)
        })
    except Exception as e:
        print(f"Aviso ao atualizar metadados da conversa {conversa_id}: {e}")

def salvar_documento_estrategico_conversa(db, conversa_id: str, documento_markdown: str):
    """Salva o documento estratégico final no documento principal da conversa e marca como concluído."""
    if not conversa_id or not documento_markdown:
        return
    try:
        db.collection('conversas').document(conversa_id).update({
            'documento_gerado': documento_markdown,
            'status': 'concluido',
            'updated_at': firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"Erro ao salvar documento estratégico no Firestore: {e}")

def atualizar_titulo_conversa(db, conversa_id: str, novo_titulo: str, prefixo_icone: str = None):
    """Atualiza o título de exibição da conversa."""
    if not conversa_id or not novo_titulo:
        return
    try:
        titulo_formatado = f"{prefixo_icone} {novo_titulo.strip()}" if prefixo_icone else novo_titulo.strip()
        db.collection('conversas').document(conversa_id).update({
            'title': titulo_formatado,
            'updated_at': firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"Erro ao atualizar o título da conversa {conversa_id}: {e}")

def renomear_conversa(db, conversa_id: str, novo_titulo: str) -> bool:
    """Renomeia o título de uma conversa no Firestore."""
    if not conversa_id or not novo_titulo:
        return False
    try:
        db.collection('conversas').document(conversa_id).update({
            'title': novo_titulo.strip(),
            'updated_at': firestore.SERVER_TIMESTAMP
        })
        return True
    except Exception as e:
        print(f"Erro ao renomear conversa {conversa_id}: {e}")
        return False

def excluir_conversa(db, conversa_id: str) -> bool:
    """Exclui permanentemente uma conversa e todas as suas mensagens do Firestore."""
    if not conversa_id:
        return False
    try:
        mensagens_ref = db.collection('conversas').document(conversa_id).collection('mensagens')
        for doc in mensagens_ref.stream():
            doc.reference.delete()
        db.collection('conversas').document(conversa_id).delete()
        return True
    except Exception as e:
        print(f"Erro ao excluir conversa {conversa_id}: {e}")
        return False

def salvar_feedback(db, conversa_id: str, user_id: str, utilidade: str, comentario: str):
    """Salva a avaliação do usuário no Firestore com metadados detalhados."""
    try:
        db.collection('feedbacks').add({
            'conversa_id': conversa_id,
            'user_id': user_id,
            'utilidade': utilidade,  # 'bom' ou 'ruim'
            'comentario': comentario,
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        return True, "Obrigado pelo seu feedback! Isso ajuda a validar a pesquisa."
    except Exception as e:
        return False, f"Erro ao salvar feedback: {e}"

