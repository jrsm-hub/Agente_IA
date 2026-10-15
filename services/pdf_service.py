import re
from fpdf import FPDF

def converter_latex_para_texto(texto: str) -> str:
    """Converte notações matemáticas/químicas em LaTeX para texto acadêmico legível no PDF."""
    if not texto or '$' not in texto:
        return texto

    # Mapeamento de símbolos gregos e comandos LaTeX comuns
    mapa_simbolos = {
        r'\Delta': 'Delta-',
        r'\delta': 'delta-',
        r'\eta': 'eta',
        r'\alpha': 'alfa',
        r'\beta': 'beta',
        r'\gamma': 'gama',
        r'\lambda': 'lambda',
        r'\mu': 'micro',
        r'\pi': 'pi',
        r'\sigma': 'sigma',
        r'\theta': 'theta',
        r'\omega': 'omega',
        r'\approx': '~',
        r'\le': '<=',
        r'\ge': '>=',
        r'\pm': '+/-',
        r'\times': 'x',
        r'\degree': '°',
        r'^\circ': '°',
        r'\circ': '°',
    }

    def limpar_expressao_math(match):
        expr = match.group(1).strip()
        
        # Substitui comandos gregos e símbolos
        for cmd, repl in mapa_simbolos.items():
            expr = expr.replace(cmd, repl)
            
        # Trata subscritos com chaves (ex: R_{ct} -> R_ct -> Rct)
        expr = re.sub(r'_\{([^\}]+)\}', r'_\1', expr)
        # Trata sobrescritos com chaves (ex: X^{2} -> X^2)
        expr = re.sub(r'\^\{([^\}]+)\}', r'^\1', expr)
        
        # Converte variáveis subscritas comuns como T_g -> Tg, R_s -> Rs, R_ct -> Rct, PCE_0 -> PCE0
        expr = re.sub(r'\b([A-Z])_([a-z0-9]{1,3})\b', r'\1\2', expr)
        expr = re.sub(r'([A-Za-z]+)_([a-zA-Z0-9]+)', r'\1_\2', expr)
        
        # Remove barras invertidas residuais
        expr = expr.replace('\\', '').strip()
        return expr

    # Converte blocos inline $...$
    texto = re.sub(r'\$([^\$\n]+)\$', limpar_expressao_math, texto)
    return texto

def corrigir_palavras_quebradas(texto: str) -> str:
    """Corrige palavras hifenizadas ou partidas acidentalmente entre quebras de linha."""
    if not texto:
        return ""
    # Corrige quebras com hífen (ex: pesqui-\nsa -> pesquisa)
    texto = re.sub(r'([a-zA-ZÀ-ÿ]+)\s*-\s*\n\s*([a-zA-ZÀ-ÿ]+)', r'\1\2', texto)
    # Corrige sufixos partidos na linha seguinte (ex: laboratório\ns -> laboratórios, térmic\na. -> térmica.)
    texto = re.sub(r'([a-zA-ZÀ-ÿ]{3,})\n([a-zà-ÿ]{1,2})\b', r'\1\2', texto)
    return texto

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
    """Padroniza e corrige marcações de negrito e destaque no texto para renderização sem asteriscos."""
    if not texto:
        return ""
    
    # 1. Unifica 3 asteriscos (***termo*** -> **termo**)
    texto = re.sub(r'\*\*\*(.*?)\*\*\*', r'**\1**', texto)
    
    # 2. Converte negrito em formato underscore (__termo__ -> **termo**)
    texto = re.sub(r'__(.*?)__', r'**\1**', texto)
    
    # 3. Corrige assimetrias comuns do LLM (ex: **termo* ou *termo**)
    texto = re.sub(r'(?<!\*)\*\*([^\*\n]+?)\*(?!\*)', r'**\1**', texto)
    texto = re.sub(r'(?<!\*)\*([^\*\n]+?)\*\*(?!\*)', r'**\1**', texto)
    
    # 4. Rótulos com dois pontos envoltos em asteriscos (ex: *Objetivo Geral:* -> **Objetivo Geral:**)
    texto = re.sub(r'(?<!\*)\*([^\*\n]+?\:)\*(?!\*)', r'**\1**', texto)
    texto = re.sub(r'(?<!\*)\*([^\*\n]+?)\*\:(?!\*)', r'**\1:**', texto)
    
    # 5. Converte ênfases de asterisco único (*termo*) em negrito para renderização limpa
    texto = re.sub(r'(?<!\*)\*([^\*\s][^\*\n]*?[^\*\s]|\w)\*(?!\*)', r'**\1**', texto)
    
    # 6. Corrige espaços indevidos dentro de marcadores (ex: ** termo ** -> **termo**)
    texto = re.sub(r'\*\*\s+(.*?)\s+\*\*', r'**\1**', texto)
    texto = re.sub(r'\*\*\s+(.*?)\*\*', r'**\1**', texto)
    texto = re.sub(r'\*\*(.*?)\s+\*\*', r'**\1**', texto)
    
    # 7. Se houver número ímpar de marcadores de negrito (tag aberta sem fechar)
    if texto.count('**') % 2 != 0:
        # Se existe um rótulo com dois pontos, fecha nos dois pontos
        if re.search(r'\*\*[^\*\n]+\:', texto):
            texto = re.sub(r'(\*\*[^\*\n]+\:)', r'\1**', texto, count=1)
        else:
            texto = texto + '**'
            
    return texto

def agrupar_blocos_markdown(texto: str) -> list[tuple[str, str]]:
    """
    Agrupa o texto markdown em blocos estruturais semânticos,
    unindo quebras de linha acidentais (soft breaks) dentro do mesmo parágrafo ou item.
    """
    linhas = texto.split('\n')
    blocos = []
    bloco_atual_tipo = None
    bloco_atual_linhas = []
    
    def salvar_bloco():
        nonlocal bloco_atual_tipo, bloco_atual_linhas
        if bloco_atual_tipo and bloco_atual_linhas:
            conteudo = " ".join(l.strip() for l in bloco_atual_linhas if l.strip())
            if conteudo:
                blocos.append((bloco_atual_tipo, conteudo))
        bloco_atual_tipo = None
        bloco_atual_linhas = []

    for linha in linhas:
        l_strip = linha.strip()
        
        # Linha em branco encerra o bloco atual
        if not l_strip:
            salvar_bloco()
            continue
            
        # Divisores horizontais
        if l_strip in ['---', '***', '___', '- - -', '* * *']:
            salvar_bloco()
            blocos.append(('divider', ''))
            continue
            
        # Delimitadores de código
        if l_strip.startswith('```'):
            salvar_bloco()
            continue

        # Cabeçalhos
        if l_strip.startswith('# '):
            salvar_bloco()
            blocos.append(('h1', l_strip[2:].strip()))
            continue
        elif l_strip.startswith('## '):
            salvar_bloco()
            blocos.append(('h2', l_strip[3:].strip()))
            continue
        elif l_strip.startswith('### '):
            salvar_bloco()
            blocos.append(('h3', l_strip[4:].strip()))
            continue

        # Item de lista numerada (ex: 1. Passo 1...)
        match_num = re.match(r'^(\d+[\.\)])\s+(.+)', l_strip)
        if match_num:
            salvar_bloco()
            bloco_atual_tipo = 'numbered:' + match_num.group(1)
            bloco_atual_linhas = [match_num.group(2)]
            continue

        # Item de lista com marcador (* ou - ou +)
        match_bullet = re.match(r'^[\*\-\+]\s+(.+)', l_strip)
        if match_bullet:
            salvar_bloco()
            bloco_atual_tipo = 'bullet'
            bloco_atual_linhas = [match_bullet.group(1)]
            continue

        # Citação em bloco (> ...)
        if l_strip.startswith('> '):
            salvar_bloco()
            bloco_atual_tipo = 'quote'
            bloco_atual_linhas = [l_strip[2:]]
            continue

        # Linha de continuação do bloco aberto
        if bloco_atual_tipo:
            bloco_atual_linhas.append(l_strip)
        else:
            bloco_atual_tipo = 'p'
            bloco_atual_linhas = [l_strip]

    salvar_bloco()
    return blocos

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

    def renderizar_bloco_formatado(self, texto_bloco, indent=0, prefixo="", altura_linha=5.5):
        """Renderiza texto com quebra de linha rigorosamente contida nas margens e suporte a negrito."""
        largura_maxima = 170 - indent  # 210mm - 40mm (margens) = 170mm
        
        texto_sanitizado = sanitizar_texto_para_pdf(texto_bloco)
        texto_sanitizado = converter_latex_para_texto(texto_sanitizado)
        texto_normalizado = normalizar_marcacoes_markdown(texto_sanitizado)
        
        # Segmenta em negrito e normal
        segmentos = re.split(r'(\*\*.*?\*\*)', texto_normalizado)
        tokens = []
        for seg in segmentos:
            if not seg:
                continue
            is_bold = seg.startswith('**') and seg.endswith('**')
            texto_seg = seg[2:-2] if is_bold else seg
            
            # Limpa asteriscos residuais
            if is_bold:
                texto_seg = texto_seg.replace('*', '').strip()
            else:
                texto_seg = re.sub(r'(^|\s)\*+(\s|$)', r'\1\2', texto_seg)
                
            palavras = texto_seg.split(' ')
            for p in palavras:
                # Remove asteriscos que tenham sobrado nas bordas da palavra
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
    
    # 1. Corrige palavras quebradas ou partidas acidentalmente
    texto_processado = corrigir_palavras_quebradas(texto_markdown)
    
    # 2. Agrupa em blocos estruturais contínuos (evita quebras de linha em parágrafos)
    blocos = agrupar_blocos_markdown(texto_processado)
    
    for tipo, conteudo in blocos:
        # Linha divisória horizontal
        if tipo == 'divider':
            pdf.ln(2.5)
            pdf.set_draw_color(226, 232, 240)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(2.5)
            continue
            
        # Título Principal (H1)
        if tipo == 'h1':
            pdf.ln(3)
            pdf.set_fill_color(241, 245, 249)  # Slate 100
            pdf.set_draw_color(30, 58, 138)    # Borda azul
            pdf.set_text_color(30, 58, 138)
            pdf.set_font("Arial", 'B', 14)
            titulo = sanitizar_texto_para_pdf(limpar_formatacao_titulo(conteudo))
            titulo = converter_latex_para_texto(titulo)
            pdf.multi_cell(170, 9, txt=f" {titulo}", border=1, fill=True, align='L')
            pdf.ln(3)
            
        # Seção (H2)
        elif tipo == 'h2':
            pdf.ln(4)
            pdf.set_text_color(30, 58, 138)  # Azul Marinho
            pdf.set_font("Arial", 'B', 12)
            subtitulo = sanitizar_texto_para_pdf(limpar_formatacao_titulo(conteudo))
            subtitulo = converter_latex_para_texto(subtitulo)
            pdf.multi_cell(170, 7, txt=subtitulo, align='L')
            pdf.set_draw_color(191, 219, 254)  # Azul claro
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(2.5)
            
        # Subseção (H3)
        elif tipo == 'h3':
            pdf.ln(2.5)
            pdf.set_text_color(51, 65, 85)   # Slate escuro
            pdf.set_font("Arial", 'B', 11)
            secao = sanitizar_texto_para_pdf(limpar_formatacao_titulo(conteudo))
            secao = converter_latex_para_texto(secao)
            pdf.multi_cell(170, 6, txt=secao, align='L')
            pdf.ln(1)
            
        # Item de Lista Numerada (ex: 1. , 2. )
        elif tipo.startswith('numbered:'):
            num = tipo.split(':', 1)[1] + " "
            pdf.renderizar_bloco_formatado(conteudo, indent=6, prefixo=num, altura_linha=5.5)
            
        # Item de Lista com marcador (bullet)
        elif tipo == 'bullet':
            pdf.renderizar_bloco_formatado(conteudo, indent=6, prefixo=f"{chr(149)} ", altura_linha=5.5)
            
        # Citação / Destaque em Bloco (> ...)
        elif tipo == 'quote':
            pdf.renderizar_bloco_formatado(conteudo, indent=8, prefixo="", altura_linha=5.5)
            
        # Parágrafo Normal
        else:
            pdf.renderizar_bloco_formatado(conteudo, indent=0, prefixo="", altura_linha=5.5)
    
    try:
        saida = pdf.output(dest='S')
    except (TypeError, ValueError):
        saida = pdf.output()

    if isinstance(saida, str):
        return saida.encode('latin-1', 'replace')
    elif isinstance(saida, (bytes, bytearray)):
        return bytes(saida)
    return bytes(saida)
