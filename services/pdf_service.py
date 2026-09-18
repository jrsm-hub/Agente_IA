import re
from fpdf import FPDF

def sanitizar_texto_para_pdf(texto: str) -> str:
    """Substitui caracteres Unicode e emojis por equivalentes compatíveis com Latin-1."""
    substituicoes = {
        '“': '"', '”': '"', '„': '"', '‟': '"',
        '‘': "'", '’': "'", '‚': "'", '‛': "'",
        '–': '-', '—': '--', '―': '--',
        '…': '...', '•': '-', '·': '-',
        '→': '->', '←': '<-', '⇒': '=>',
        '©': '(c)', '®': '(R)', '™': '(TM)',
        '🧭': '', '📚': '', '✍': '', '💡': '', '🚀': '', '⭐': '', '👍': '', '👎': '',
        '👉': '>', '📌': '*', '📍': '*', '✅': '[OK]', '❌': '[X]', '⚠️': '[!]'
    }
    for de, para in substituicoes.items():
        texto = texto.replace(de, para)
    return texto.encode('latin-1', 'replace').decode('latin-1')

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
        
        # Segmenta em negrito e normal
        segmentos = re.split(r'(\*\*.*?\*\*)', texto_sanitizado)
        tokens = []
        for seg in segmentos:
            if not seg:
                continue
            is_bold = seg.startswith('**') and seg.endswith('**')
            texto_seg = seg[2:-2] if is_bold else seg
            palavras = texto_seg.split(' ')
            for p in palavras:
                if p:
                    tokens.append((p, is_bold))
                    
        if not tokens:
            self.ln(altura_linha)
            return

        pos_x_inicio = self.l_margin + indent
        self.set_x(pos_x_inicio)
        
        if prefixo:
            self.set_font('Arial', 'B', 10)
            self.set_text_color(37, 99, 235)
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
    pdf = AcademicPDF(orientation='P', unit='mm', format='A4')
    pdf.set_margins(left=20, top=18, right=20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.alias_nb_pages()
    pdf.add_page()
    
    linhas = texto_markdown.split('\n')
    
    for linha in linhas:
        linha_sanitizada = sanitizar_texto_para_pdf(linha)
        linha_strip = linha_sanitizada.strip()
        
        # Título Principal (H1)
        if linha_strip.startswith('# '):
            pdf.ln(3)
            pdf.set_fill_color(241, 245, 249)  # Slate 100
            pdf.set_draw_color(30, 58, 138)    # Borda azul
            pdf.set_text_color(30, 58, 138)
            pdf.set_font("Arial", 'B', 14)
            titulo = linha_strip[2:].strip()
            pdf.multi_cell(170, 9, txt=f" {titulo}", border=1, fill=True, align='L')
            pdf.ln(3)
            
        # Seção (H2)
        elif linha_strip.startswith('## '):
            pdf.ln(4)
            pdf.set_text_color(30, 58, 138)  # Azul Marinho
            pdf.set_font("Arial", 'B', 12)
            subtitulo = linha_strip[3:].strip()
            pdf.multi_cell(170, 7, txt=subtitulo, align='L')
            pdf.set_draw_color(191, 219, 254)  # Azul claro
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(2.5)
            
        # Subseção (H3)
        elif linha_strip.startswith('### '):
            pdf.ln(2.5)
            pdf.set_text_color(51, 65, 85)   # Slate escuro
            pdf.set_font("Arial", 'B', 11)
            secao = linha_strip[4:].strip()
            pdf.multi_cell(170, 6, txt=secao, align='L')
            pdf.ln(1)
            
        # Item de Lista (* ou -)
        elif linha_strip.startswith('* ') or linha_strip.startswith('- '):
            conteudo_item = linha_strip[2:].strip()
            pdf.renderizar_bloco_formatado(conteudo_item, indent=6, prefixo=f"{chr(149)} ", altura_linha=5.5)
            
        # Linha Vazia
        elif not linha_strip:
            pdf.ln(2)
            
        # Parágrafo Normal
        else:
            pdf.renderizar_bloco_formatado(linha_strip, indent=0, prefixo="", altura_linha=5.5)
    
    return pdf.output(dest='S').encode('latin-1')
