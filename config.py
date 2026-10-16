import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Função para localização robusta da chave em qualquer formato do Streamlit Secrets
def obter_google_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key:
        if isinstance(key, (list, tuple)) and key:
            key = key[0]
        if str(key).strip():
            return str(key).strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GOOGLE_API_KEY" in st.secrets:
                val = st.secrets["GOOGLE_API_KEY"]
                if isinstance(val, (list, tuple)) and val:
                    val = val[0]
                return str(val).strip()
            if "GEMINI_API_KEY" in st.secrets:
                val = st.secrets["GEMINI_API_KEY"]
                if isinstance(val, (list, tuple)) and val:
                    val = val[0]
                return str(val).strip()
            # Varredura profunda em todas as seções (ex: caso colado ao final após [firebase_credentials])
            for sec_name in st.secrets:
                sec_val = st.secrets[sec_name]
                if isinstance(sec_val, dict) or hasattr(sec_val, "get"):
                    sub_key = sec_val.get("GOOGLE_API_KEY") or sec_val.get("GEMINI_API_KEY")
                    if sub_key:
                        if isinstance(sub_key, (list, tuple)) and sub_key:
                            sub_key = sub_key[0]
                        if str(sub_key).strip():
                            return str(sub_key).strip()
    except Exception:
        pass
    return None

GOOGLE_API_KEY = obter_google_api_key()
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Modelos com fallback e suporte a secrets
LLM_MODEL = os.getenv("LLM_MODEL")
if not LLM_MODEL:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "LLM_MODEL" in st.secrets:
            val_model = st.secrets["LLM_MODEL"]
            if isinstance(val_model, (list, tuple)) and val_model:
                val_model = val_model[0]
            LLM_MODEL = str(val_model).strip()
    except Exception:
        pass
DEPRECATED_MODELS_MAP = {
    "gemini-2.5-flash": "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite": "gemini-3.5-flash-lite",
    "gemini-1.5-flash": "gemini-3.5-flash-lite",
    "gemini-1.5-pro": "gemini-3.5-flash-lite",
    "models/gemini-2.5-flash": "gemini-3.5-flash-lite",
    "models/gemini-2.5-flash-lite": "gemini-3.5-flash-lite",
    "models/gemini-1.5-flash": "gemini-3.5-flash-lite",
}

if not LLM_MODEL or LLM_MODEL in DEPRECATED_MODELS_MAP:
    LLM_MODEL = DEPRECATED_MODELS_MAP.get(LLM_MODEL, "gemini-3.5-flash-lite")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))

# Diretórios e Caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONTES_DIR = os.path.join(BASE_DIR, "fontes")
TEXTOS_PROCESSADOS_DIR = os.path.join(BASE_DIR, "textos_processados")
BANCO_VETORIAL_DIR = os.path.join(BASE_DIR, "banco_vetorial_chroma")
FRAGMENTOS_PKL_PATH = os.path.join(BASE_DIR, "fragmentos.pkl")
FIREBASE_CREDENTIALS_PATH = os.path.join(BASE_DIR, "firebase-credentials.json")
