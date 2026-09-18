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

def criar_conversa(db, user_id: str, mensagem_inicial: str, tipo: str = "estrategia") -> str:
    """Cria uma nova conversa vinculada a um usuário e insere a mensagem inicial."""
    nova_conversa_ref = db.collection('conversas').document()
    conversa_id = nova_conversa_ref.id
    
    icone = "🧭" if tipo == "estrategia" else "📚"
    titulo_padrao = f"{icone} Conversa {conversa_id[:4]}..."
    
    nova_conversa_ref.set({
        'user_id': user_id,
        'tipo': tipo,
        'timestamp': firestore.SERVER_TIMESTAMP,
        'title': titulo_padrao
    })
    
    nova_conversa_ref.collection('mensagens').add({
        'role': 'assistant',
        'content': mensagem_inicial,
        'timestamp': firestore.SERVER_TIMESTAMP
    })
    
    return conversa_id

def listar_conversas(db, user_id: str = None):
    """Retorna a lista de conversas do usuário ordenadas pela data mais recente."""
    if user_id:
        # Busca filtrada por usuário e ordena em memória para evitar necessidade de índices compostos
        docs = db.collection('conversas').where('user_id', '==', user_id).stream()
    else:
        docs = db.collection('conversas').order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
    
    lista = list(docs)
    # Ordena pelo timestamp decrescente (mais recente primeiro)
    lista.sort(
        key=lambda d: (d.to_dict().get('timestamp') or 0),
        reverse=True
    )
    return lista

def carregar_mensagens_conversa(db, conversa_id: str):
    """Carrega todas as mensagens de uma conversa ordenadas por timestamp."""
    msgs_db = db.collection('conversas').document(conversa_id).collection('mensagens').order_by('timestamp').stream()
    return [msg.to_dict() for msg in msgs_db]

def obter_dados_conversa(db, conversa_id: str) -> dict:
    """Retorna os metadados da conversa (título, tipo, user_id)."""
    if not conversa_id:
        return {}
    doc = db.collection('conversas').document(conversa_id).get()
    return doc.to_dict() if doc.exists else {}

def adicionar_mensagem(db, conversa_id: str, role: str, content: str):
    """Salva uma nova mensagem no histórico da conversa no Firestore."""
    if not conversa_id:
        return
    db.collection('conversas').document(conversa_id).collection('mensagens').add({
        'role': role,
        'content': content,
        'timestamp': firestore.SERVER_TIMESTAMP
    })

def atualizar_titulo_conversa(db, conversa_id: str, novo_titulo: str, prefixo_icone: str = None):
    """Atualiza o título de exibição da conversa."""
    if not conversa_id or not novo_titulo:
        return
    try:
        titulo_formatado = f"{prefixo_icone} {novo_titulo.strip()}" if prefixo_icone else novo_titulo.strip()
        db.collection('conversas').document(conversa_id).update({'title': titulo_formatado})
    except Exception as e:
        print(f"Erro ao atualizar o título da conversa {conversa_id}: {e}")

def renomear_conversa(db, conversa_id: str, novo_titulo: str) -> bool:
    """Renomeia o título de uma conversa no Firestore."""
    if not conversa_id or not novo_titulo:
        return False
    try:
        db.collection('conversas').document(conversa_id).update({'title': novo_titulo.strip()})
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
    """Salva a avaliação do usuário no Firestore."""
    try:
        db.collection('feedbacks').add({
            'conversa_id': conversa_id,
            'user_id': user_id,
            'utilidade': utilidade,  # 'bom' ou 'ruim'
            'comentario': comentario,
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        return True, "Obrigado pelo seu feedback! Isso ajuda a melhorar a pesquisa."
    except Exception as e:
        return False, f"Erro ao salvar feedback: {e}"

