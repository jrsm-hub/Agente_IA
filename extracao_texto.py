# 1_extracao_texto.py 

import fitz  # PyMuPDF
import os
import unicodedata # <<< NOVA IMPORTAÇÃO: Biblioteca para normalizar texto

# Caminho para a pasta onde estão os seus PDFs
pasta_fontes = "fontes"
# Caminho para a pasta onde salvaremos os textos extraídos
pasta_textos = "textos_processados"

# Cria a pasta de destino se ela não existir
if not os.path.exists(pasta_textos):
    os.makedirs(pasta_textos)

print("Iniciando a extração de texto dos PDFs...")

# Lista todos os arquivos na pasta de fontes
for nome_arquivo_pdf in os.listdir(pasta_fontes):
    # Verifica se o arquivo é um PDF
    if nome_arquivo_pdf.endswith(".pdf"):
        caminho_pdf = os.path.join(pasta_fontes, nome_arquivo_pdf)

        try:
            # Abre o arquivo PDF
            doc = fitz.open(caminho_pdf)
            texto_completo = ""

            # Itera sobre cada página do PDF
            for pagina in doc:
                texto_completo += pagina.get_text()

          
            # Normaliza o texto para o formato NFC (Normalization Form C)
            # Este formato combina os caracteres com seus acentos em um único ponto de código.
            # Exemplo: 'c' + '¸' vira 'ç'
            texto_normalizado = unicodedata.normalize("NFC", texto_completo)
           

            # Cria um nome para o arquivo de texto de saída
            nome_arquivo_txt = nome_arquivo_pdf.replace(".pdf", ".txt")
            caminho_txt = os.path.join(pasta_textos, nome_arquivo_txt)

            # Salva o texto NORMALIZADO em um novo arquivo .txt
            with open(caminho_txt, "w", encoding="utf-8") as f:
                f.write(texto_normalizado) # <<< ALTERADO para escrever o texto normalizado

            print(f"Texto extraído e normalizado de '{nome_arquivo_pdf}' e salvo em '{nome_arquivo_txt}'")

        except Exception as e:
            print(f"Erro ao processar o arquivo {nome_arquivo_pdf}: {e}")

print("\nExtração de texto concluída com sucesso!")