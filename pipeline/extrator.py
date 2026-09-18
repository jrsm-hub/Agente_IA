import os
import fitz  # PyMuPDF
import unicodedata
from config import FONTES_DIR, TEXTOS_PROCESSADOS_DIR

def extrair_textos_pdf(pasta_fontes=FONTES_DIR, pasta_saida=TEXTOS_PROCESSADOS_DIR):
    """Extrai e normaliza o texto de todos os PDFs da pasta fontes."""
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)

    if not os.path.exists(pasta_fontes):
        print(f"Pasta de fontes '{pasta_fontes}' não encontrada.")
        return []

    arquivos_processados = []
    print("Iniciando a extração de texto dos PDFs...")

    for nome_arquivo in os.listdir(pasta_fontes):
        if nome_arquivo.lower().endswith(".pdf"):
            caminho_pdf = os.path.join(pasta_fontes, nome_arquivo)
            try:
                doc = fitz.open(caminho_pdf)
                texto_completo = ""

                for pagina in doc:
                    texto_completo += pagina.get_text()

                texto_normalizado = unicodedata.normalize("NFC", texto_completo)
                nome_arquivo_txt = os.path.splitext(nome_arquivo)[0] + ".txt"
                caminho_txt = os.path.join(pasta_saida, nome_arquivo_txt)

                with open(caminho_txt, "w", encoding="utf-8") as f:
                    f.write(texto_normalizado)

                print(f"Texto extraído: '{nome_arquivo}' -> '{nome_arquivo_txt}'")
                arquivos_processados.append(caminho_txt)
            except Exception as e:
                print(f"Erro ao processar '{nome_arquivo}': {e}")

    print(f"Extração concluída! {len(arquivos_processados)} arquivos processados.")
    return arquivos_processados

if __name__ == "__main__":
    extrair_textos_pdf()
