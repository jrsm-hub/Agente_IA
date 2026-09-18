import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Chaves de API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Modelos
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))

# Diretórios e Caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONTES_DIR = os.path.join(BASE_DIR, "fontes")
TEXTOS_PROCESSADOS_DIR = os.path.join(BASE_DIR, "textos_processados")
BANCO_VETORIAL_DIR = os.path.join(BASE_DIR, "banco_vetorial_chroma")
FRAGMENTOS_PKL_PATH = os.path.join(BASE_DIR, "fragmentos.pkl")
FIREBASE_CREDENTIALS_PATH = os.path.join(BASE_DIR, "firebase-credentials.json")
