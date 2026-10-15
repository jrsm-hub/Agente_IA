import re
from fpdf import FPDF

def sanitizar_texto_para_pdf(texto) -> str:
    """Substitui caracteres Unicode e emojis por equivalentes compatíveis com Latin-1."""
    if not isinstance(texto, str):
        from services.ai_service import extrair_texto_conteudo
        texto = extrair_texto_conteudo(texto)
    substituicoes = {
        '“': '"', '”': '"', '„': '"', '‟': '"',
        '‘': "'", '’': "'", '‚': "'", '‛': "'",
        '–': '-', '—': '--', '―': '--',
        '…': '...', '•': '-', '·': '-',
        '→': '->', '←': '<-', '⇒': '=>',
        '©': '(c)', '®': '(R)', '™': '(TM)',
        '🧭': '', '📚': '', '✍': '', '💡': '', '🚀': '', '⭐': '', '👍': '', '👎': '',
        '👉': '>', '📌': '', '📍': '', '✅': '[OK]', '❌': '[X]', '⚠️': '[!]'
    }
    for de, para in substituicoes.items():
        texto = texto.replace(de, para)
    return texto.encode('latin-1', 'replace').decode('latin-1')

def limpar_formatacao_titulo(texto: str) -> str:
    """Remove marcações de markdown residuais (*, **, __, #) em cabeçalhos."""
    if not texto:
        return ""
    t = texto.strip()
    # Remove marcações de negrito e itálico que não fazem sentido em títulos já negritados
    t = t.replace('***', '').replace('**', '').replace('__', '')
    # Remove asteriscos ou underscores nos limites das palavras
    t = re.sub(r'(?<!\w)\*+([^\*]+?)\*+(?!\w)', r'\1', t)
    t = re.sub(r'(?<!\w)_+([^_]+?)_+(?!\w)', r'\1', t)
    # Remove qualquer outro asterisco remanescente
    t = t.replace('*', '').strip()
    return t

def normalizar_marcacoes_markdown(texto: str) -> str:
    """Padroniza e corrige marcações de negrito e destaque no texto."""
    if not texto:
        return ""
    
    # 1. Unifica 3 asteriscos (***termo*** -> **termo**)
    texto = re.sub(r'\*\*\*(.*?)\*\*\*', r'**\1**', texto)
    
    # 2. Converte negrito em formato underscore (__termo__ -> **termo**)
    texto = re.sub(r'__(.*?)__', r'**\1**', texto)
    
    # 3. Corrige assimetrias comuns do LLM (ex: **termo* ou *termo**)
    texto = re.sub(r'(?<!\*)\*\*([^\*\n]+?)\*(?!\*)', r'**\1**', texto)
    texto = re.sub(r'(?<!\*)\*([^\*\n]+?)\*\*(?!\*)', r'**\1**', texto)
    
    # 4. Converte ênfases de asterisco único (*termo*) em negrito para renderização correta
    texto = re.sub(r'(?<!\*)\*([^\*\s][^\*\n]*?[^\*\s]|\w)\*(?!\*)', r'**\1**', texto)
    
    # 5. Corrige espaços indevidos dentro de marcadores (ex: ** termo ** -> **termo**)
    texto = re.sub(r'\*\*\s+(.*?)\s+\*\*', r'**\1**', texto)
    texto = re.sub(r'\*\*\s+(.*?)\*\*', r'**\1**', texto)
    texto = re.sub(r'\*\*(.*?)\s+\*\*', r'**\1**', texto)
    
    # 6. Se houver número ímpar de marcadores de negrito (tag aberta sem fechar)
    if texto.count('**') % 2 != 0:
        # Se existe um rótulo com dois pontos (ex: **Subárea / Tópico: resto...), fecha nos dois pontos
        if re.search(r'\*\*[^\*\n]+\:', texto):
            texto = re.sub(r'(\*\*[^\*\n]+\:)', r'\1**', texto, count=1)
        else:
            # Caso contrário, fecha no final
            texto = texto + '**'
            
    return texto

class AcademicPDF(FPDF):
    """Classe especializada para relatórios acadêmicos estruturados e elegantes."""
    
    def header(self):
        # Barra superior decorativa
        self.set_fill_color(30, 58, 138)  # Azul Marinho Acadêmico
        self.rect(0, 0, 210, 4, 'F')
        
        # Cabeçalho da página
        self.set_font("Arial", 'B', 8)
        self.set_text_color(100, 116, 139)  # Cinza ardósia
        self.cell(0, 6, "ESTRATEGISTA DE PESQUISA ACADÊMICA | RELATÓRIO DE DIRETRIZES", ln=True, align='R')
        
        # Linha divisória sutil
        self.set_draw_color(226, 232, 240)
        self.line(20, 12, 190, 12)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(226, 232, 240)
        self.line(20, 282, 190, 282)
        
        self.set_font("Arial", 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(100, 10, "Documento Estratégico de Pesquisa - Gerado por Inteligência Artificial", align='L')
        self.cell(0, 10, f"Página {self.page_no()}/{{nb}}", align='R')

    def renderizar_bloco_formatado(self, texto_sanitizado, indent=0, prefixo="", altura_linha=5.5):
        """Renderiza texto com quebra de linha rigorosamente contida nas margens e suporte a negrito."""
        largura_maxima = 170 - indent  # 210mm - 40mm (margens) = 170mm
        
        texto_normalizado = normalizar_marcacoes_markdown(texto_sanitizado)
        
        # Segmenta em negrito e normal
        segmentos = re.split(r'(\*\*.*?\*\*)', texto_normalizado)
        tokens = []
        for seg in segmentos:
            if not seg:
                continue
            is_bold = seg.startswith('**') and seg.endswith('**')
            texto_seg = seg[2:-2] if is_bold else seg
            
            # Remove qualquer asterisco residual de marcações malformadas
            if is_bold:
                texto_seg = texto_seg.replace('*', '').strip()
            else:
                # Remove asteriscos soltos que não fazem parte do texto
                texto_seg = re.sub(r'(^|\s)\*+(\s|$)', r'\1\2', texto_seg)
                
            palavras = texto_seg.split(' ')
            for p in palavras:
                # Remove asteriscos residuais nas bordas da palavra
                p_limpo = re.sub(r'^\*+|\*+$', '', p)
                if p_limpo:
                    tokens.append((p_limpo, is_bold))
                    
        if not tokens:
            self.ln(altura_linha)
            return

        pos_x_inicio = self.l_margin + indent
        self.set_x(pos_x_inicio)
        
        if prefixo:
            self.set_font('Arial', 'B', 10)
            self.set_text_color(30, 58, 138)  # Azul Marinho Acadêmico
            self.write(altura_linha, prefixo)
            largura_acumulada = self.get_string_width(prefixo)
        else:
            largura_acumulada = 0

        for i, (palavra, is_bold) in enumerate(tokens):
            self.set_font('Arial', 'B' if is_bold else '', 10)
            self.set_text_color(15, 23, 42) if is_bold else self.set_text_color(51, 65, 85)
            
            espaco = " " if i < len(tokens) - 1 else ""
            texto_token = palavra + espaco
            largura_token = self.get_string_width(texto_token)
            
            # Se o token estourar a largura da página, pula para a próxima linha respeitando a indentação
            if largura_acumulada + largura_token > largura_maxima and largura_acumulada > 0:
                self.ln(altura_linha)
                self.set_x(pos_x_inicio)
                largura_acumulada = 0
                
            self.write(altura_linha, texto_token)
            largura_acumulada += largura_token
            
        self.ln(altura_linha)

def criar_pdf_formatado(texto_markdown: str) -> bytes:
    """Gera um documento PDF com design acadêmico premium e controle rigoroso de margens."""
    if not texto_markdown:
        return b""
        
    pdf = AcademicPDF(orientation='P', unit='mm', format='A4')
    pdf.set_margins(left=20, top=18, right=20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Pré-processamento: unifica quebras de linha acidentais dentro de marcações de negrito
    texto_markdown = re.sub(r'\*\*([^\*\n]+)\n([^\*\n]+)\*\*', r'**\1 \2**', texto_markdown)
    
    linhas = texto_markdown.split('\n')
    
    for linha in linhas:
        linha_sanitizada = sanitizar_texto_para_pdf(linha)
        linha_strip = linha_sanitizada.strip()
        
        # Linha em branco
        if not linha_strip:
            pdf.ln(2)
            continue
            
        # Linha divisória horizontal (--- ou *** ou ___)
        if linha_strip in ['---', '***', '___', '- - -', '* * *']:
            pdf.ln(2.5)
            pdf.set_draw_color(226, 232, 240)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(2.5)
            continue
            
        # Ignora delimitadores de código markdown (```)
        if linha_strip.startswith('```'):
            continue
            
        # Título Principal (H1)
        if linha_strip.startswith('# '):
            pdf.ln(3)
            pdf.set_fill_color(241, 245, 249)  # Slate 100
            pdf.set_draw_color(30, 58, 138)    # Borda azul
            pdf.set_text_color(30, 58, 138)
            pdf.set_font("Arial", 'B', 14)
            titulo = limpar_formatacao_titulo(linha_strip[2:])
            pdf.multi_cell(170, 9, txt=f" {titulo}", border=1, fill=True, align='L')
            pdf.ln(3)
            
        # Seção (H2)
        elif linha_strip.startswith('## '):
            pdf.ln(4)
            pdf.set_text_color(30, 58, 138)  # Azul Marinho
            pdf.set_font("Arial", 'B', 12)
            subtitulo = limpar_formatacao_titulo(linha_strip[3:])
            pdf.multi_cell(170, 7, txt=subtitulo, align='L')
            pdf.set_draw_color(191, 219, 254)  # Azul claro
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(2.5)
            
        # Subseção (H3)
        elif linha_strip.startswith('### '):
            pdf.ln(2.5)
            pdf.set_text_color(51, 65, 85)   # Slate escuro
            pdf.set_font("Arial", 'B', 11)
            secao = limpar_formatacao_titulo(linha_strip[4:])
            pdf.multi_cell(170, 6, txt=secao, align='L')
            pdf.ln(1)
            
        # Item de Lista Numerada (ex: 1. **Passo 1:** ...)
        elif re.match(r'^\d+\.\s+', linha_strip):
            match_num = re.match(r'^(\d+\.)\s+(.+)', linha_strip)
            if match_num:
                num_prefixo = match_num.group(1) + " "
                conteudo_item = match_num.group(2).strip()
                pdf.renderizar_bloco_formatado(conteudo_item, indent=6, prefixo=num_prefixo, altura_linha=5.5)
            else:
                pdf.renderizar_bloco_formatado(linha_strip, indent=0, prefixo="", altura_linha=5.5)
            
        # Item de Lista com marcador (* ou - ou +)
        elif linha_strip.startswith('* ') or linha_strip.startswith('- ') or linha_strip.startswith('+ '):
            conteudo_item = linha_strip[2:].strip()
            pdf.renderizar_bloco_formatado(conteudo_item, indent=6, prefixo=f"{chr(149)} ", altura_linha=5.5)
            
        # Citação / Destaque em Bloco (> ...)
        elif linha_strip.startswith('> '):
            conteudo_quote = linha_strip[2:].strip()
            pdf.renderizar_bloco_formatado(conteudo_quote, indent=8, prefixo="", altura_linha=5.5)
            
        # Parágrafo Normal
        else:
            pdf.renderizar_bloco_formatado(linha_strip, indent=0, prefixo="", altura_linha=5.5)
    
    try:
        saida = pdf.output(dest='S')
    except (TypeError, ValueError):
        saida = pdf.output()

    if isinstance(saida, str):
        return saida.encode('latin-1', 'replace')
    elif isinstance(saida, (bytes, bytearray)):
        return bytes(saida)
    return bytes(saida)

