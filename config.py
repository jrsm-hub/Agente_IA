import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Chaves de API: busca em os.getenv, st.secrets (Streamlit Cloud) e sincroniza com o ambiente
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GOOGLE_API_KEY" in st.secrets:
                GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
            elif "GEMINI_API_KEY" in st.secrets:
                GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
            elif "general" in st.secrets and "GOOGLE_API_KEY" in st.secrets["general"]:
                GOOGLE_API_KEY = st.secrets["general"]["GOOGLE_API_KEY"]
    except Exception:
        pass

if GOOGLE_API_KEY:
    GOOGLE_API_KEY = str(GOOGLE_API_KEY).strip()
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Modelos com fallback e suporte a secrets
LLM_MODEL = os.getenv("LLM_MODEL")
if not LLM_MODEL:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "LLM_MODEL" in st.secrets:
            LLM_MODEL = st.secrets["LLM_MODEL"]
    except Exception:
        pass
if not LLM_MODEL:
    LLM_MODEL = "gemini-flash-lite-latest"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))

# Diretórios e Caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONTES_DIR = os.path.join(BASE_DIR, "fontes")
TEXTOS_PROCESSADOS_DIR = os.path.join(BASE_DIR, "textos_processados")
BANCO_VETORIAL_DIR = os.path.join(BASE_DIR, "banco_vetorial_chroma")
FRAGMENTOS_PKL_PATH = os.path.join(BASE_DIR, "fragmentos.pkl")
FIREBASE_CREDENTIALS_PATH = os.path.join(BASE_DIR, "firebase-credentials.json")
