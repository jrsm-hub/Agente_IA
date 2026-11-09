# CAPÍTULO 4: O ARTEFATO — ARQUITETURA E IMPLEMENTAÇÃO DO AGENTE INTELIGENTE

Este capítulo descreve a concepção, arquitetura de software e detalhes de implementação do artefato computacional desenvolvido nesta pesquisa, em consonância com a etapa de *Design and Development* da metodologia *Design Science Research* (DSR) (VAISHNAVI; KUECHLER, 2015). O sistema, denominado **Orientador Acadêmico Inteligente**, é um software de código aberto, concebido sob uma arquitetura modular desacoplada em Python, integrando modelos de linguagem de grande escala (LLMs), um orquestrador socrático estruturado e um subsistema de Geração Aumentada por Recuperação (RAG) ancorado na literatura canônica de metodologia científica.

---

## 4.1 Visão Geral e Requisitos do Sistema

O objetivo central do artefato é mitigar as barreiras cognitivas e metodológicas enfrentadas por estudantes de graduação na fase de ideação do Trabalho de Conclusão de Curso (TCC), guiando-os desde uma área genérica de interesse até a formalização de uma pergunta científica testável e de um plano de execução metodológica fundamentado.

Para orientar o desenvolvimento do artefato, foram estabelecidos requisitos funcionais e não funcionais alinhados aos preceitos pedagógicos da mediação socrática e da reprodutibilidade científica.

### 4.1.1 Requisitos Funcionais (RF)
* **RF01 - Condução Dialógica Socrática:** O sistema deve conduzir uma entrevista em turnos estruturados, formulando perguntas reflexivas de afunilamento temático a partir das contribuições progressivas do usuário.
* **RF02 - Sugestão Metodológica em Múltipla Escolha:** A cada rodada do questionário socrático, o agente deve fornecer exatamente 4 alternativas concretas e metodologicamente viáveis de aprofundamento, permitindo também a inserção de respostas dissertativas livres.
* **RF03 - Síntese do Documento Estratégico:** Ao concluir a coleta de ideação, o sistema deve compilar todo o histórico dialógico em um Documento Estratégico formal em Markdown contendo: diagnóstico do tema, pergunta norteadora, duas abordagens metodológicas comparativas (Caminho A e Caminho B), riscos/mitigações, recomendações e cronograma estimado por etapas.
* **RF04 - Exportação em Formato Tipográfico Acadêmico (PDF):** O artefato deve compilar o Documento Estratégico em um relatório PDF diagramado segundo padrões estéticos formais (margens de 20 mm, tipografia legível, cabeçalho e rodapé institucionais).
* **RF05 - Diálogo Metodológico Aberto com RAG:** Após a emissão do documento, o sistema deve manter uma sessão conversacional aberta, consultando uma base vetorial de manuais de pesquisa para esclarecer dúvidas procedimentais (amostragem, métricas, delineamento experimental).
* **RF06 - Modo Consultor Metodológico Estrito e Auditoria de Rascunhos:** O sistema deve disponibilizar um módulo independente com temperatura zero e verificação estrita anti-alucinação, permitindo a submissão de textos/rascunhos em PDF ou TXT para emissão de parecer fundamentado com citação formal das obras consultadas.
* **RF07 - Persistência e Rastreabilidade Multi-usuário:** As sessões de atendimento, dados de interação, escolhas socráticas, rascunhos gerados e avaliações de feedback devem ser persistidos individualmente na nuvem, vinculados ao identificador institucional do discente.

### 4.1.2 Requisitos Não Funcionais (RNF)
* **RNF01 - Baixa Latência e Streaming Nativo:** As respostas socráticas e os fluxos RAG devem iniciar a exibição textual para o discente em menos de 2 segundos, utilizando *token streaming* em tempo real (`st.write_stream`).
* **RNF02 - Mitigação de Ruído Conceitual:** O módulo socrático não deve realizar buscas vetoriais indiscriminadas durante a ideação inicial, prevenindo que terminologias metodológicas genéricas interfiram prematuramente no fluxo criativo de tópicos técnicos específicos (ex: redes, visão computacional, sistemas distribuídos).
* **RNF03 - Rigor Normativo e Prevenção de Alucinações:** As respostas metodológicas no modo consultor devem basear-se estritamente nos fragmentos recuperados da base de conhecimento, recusando afirmações sem respaldo na literatura disponível.
* **RNF04 - Portabilidade e Acesso em Nuvem:** A aplicação deve ser plenamente executável via navegador web através do Streamlit Community Cloud ou servidor local dedicado, sem exigir placas aceleradoras (GPU) nos dispositivos dos estudantes.

---

## 4.2 Arquitetura Geral da Solução

O sistema adota uma arquitetura em camadas desacopladas (*Layered Architecture*), dividida em três pilares principais: **Camada de Apresentação e Interação**, **Camada de Cognição e Serviços de IA/RAG**, e **Camada de Persistência e Dados**.

```
┌────────────────────────────────────────────────────────────────────────┐
│               CAMADA DE APRESENTAÇÃO (app.py - Streamlit)              │
│  - Autenticação Simples / Identificador (@cin.ufpe.br)                 │
│  - Seletor de Modo: [Estratégia Socrática] vs. [Consultoria RAG]       │
│  - Chat Conversacional com Streaming Nativo (st.write_stream)          │
│  - Botões Rápidos de Decisão Metodológica (1 a 4)                      │
│  - Gerador e Download de PDF Acadêmico (FPDF)                          │
│  - Painel de Histórico e Atendimentos Salvos (Sidebar)                 │
└───────────────────┬───────────────────────────────┬────────────────────┘
                    │                               │
┌───────────────────▼──────────────┐   ┌────────────▼────────────────────┐
│   CAMADA DE COGNIÇÃO E SERVIÇOS  │   │   CAMADA DE PERSISTÊNCIA/DADOS  │
│  - ai_service.py:                │   │  - firebase_service.py:         │
│    • Orquestrador Socrático      │   │    • Google Firestore NoSQL     │
│    • Síntese Estratégica         │   │    • Coleção 'conversas'        │
│    • Diálogo Pós-Estratégia      │   │    • Subcoleção 'mensagens'     │
│    • Consultor RAG Estrito (0.0) │   │    • Telemetria & Avaliações    │
│    • LLM Fallback (gemini-flash) │   │  - banco_vetorial_chroma/:      │
│  - rag_service.py:               │   │    • ChromaDB Local Persistente │
│    • Similarity Search (k=3, k=4)│   │    • 5.839 Chunks Semânticos    │
│  - pdf_service.py:               │   │    • gemini-embedding-001       │
│    • Normalizador Tipográfico    │   │  - fontes/:                     │
│    • Engine Anti-Quebra FPDF     │   │    • 21 Manuais de Metodologia  │
└──────────────────────────────────┘   └─────────────────────────────────┘
```

A independência modular garante que a camada de inteligência (`ai_service.py`) opere de forma transparente em relação ao motor de interface (`app.py`), facilitando substituições de modelos de linguagem ou migrações de provedor sem impacto no fluxo de usuário.

---

## 4.3 Módulo Dialógico Socrático: Ideação e Afunilamento Temático

A ideação científica é frequentemente descrita pelos estudantes como a etapa de maior estresse cognitivo, caracterizada pelo bloqueio da "página em branco" e pela imprecisão do escopo investigativo. O Módulo Dialógico Socrático foi projetado para atuar como um mediador pedagógico ativo, decompondo a complexidade do afunilamento temático em decisões graduais de baixo atrito.

### 4.3.1 Desacoplamento Cognitivo: RAG vs. Diálogo Socrático
Uma decisão de projeto fundamental consistiu no **desacoplamento deliberado** entre o questionário socrático inicial e a busca vetorial RAG:
1. **Redução de Latência:** Uma consulta vetorial seguida de re-ranqueamento a cada turno dialógico introduziria latência desnecessária (de 3 a 5 segundos). O uso de engenharia de prompts direcionada sobre o histórico acumulado mantém o tempo de resposta em ~1,2 segundo.
2. **Prevenção de Ruído Temático:** Durante o afunilamento de tópicos altamente técnicos de Ciência da Computação (como arquitetura de microserviços, visão computacional ou computação quântica), a injeção indiscriminada de fragmentos de livros clássicos de metodologia (ex: técnicas de questionário de Lakatos ou Gil) atua como ruído cognitivo no contexto do LLM, desviando as alternativas de caminhos específicos da computação.

### 4.3.2 Engenharia de Prompts do Orquestrador Socrático
O agente socrático opera com base em um *System Prompt* rigorosamente parametrizado (`stream_gerar_proxima_pergunta`), instruído a analisar o histórico da conversa e gerar exatamente uma pergunta norteadora e 4 opções concretas:

```text
Você é um orientador de pesquisa experiente e a sua missão é conduzir uma entrevista socrática para ajudar um aluno a delimitar seu projeto de pesquisa acadêmica, guiando-o desde uma área ampla até um tópico específico, rigoroso e exequível.

Regras:
1. Analise o histórico completo da conversa para entender o contexto atual e a última contribuição do aluno.
2. Formule a próxima pergunta de forma clara, reflexiva e direta em português do Brasil, estimulando o afunilamento temático.
3. Ofereça exatamente 4 opções metodologicamente consistentes, inovadoras e acessíveis de aprofundamento.
4. Formate as opções rigorosamente como lista numerada de 1 a 4, cada qual em uma linha separada.
5. Tom didático e acessível: Mantenha um tom encorajador, claro e objetivo. Evite jargões herméticos ou linguagem excessivamente rebuscada, preservando a seriedade e o rigor acadêmico sem emojis ou gírias.
6. Responda estritamente com a pergunta seguida das 4 opções, sem saudações redundantes ou introduções.
```

### 4.3.3 Dinâmica de Interação e Suporte a Respostas Livres
Na interface Streamlit, a função `separar_pergunta_e_opcoes()` decompõe a saída do modelo em duas componentes:
- O texto explicativo/pergunta (exibido na bolha conversacional do orientador);
- As 4 alternativas (renderizadas como botões interativos dinâmicos de largura total).

O estudante pode avançar na entrevista clicando diretamente em uma das opções sugeridas ou digitando uma resposta personalizada no campo de entrada (`st.chat_input`), garantindo flexibilidade caso queira direcionar a pesquisa para um contexto específico.

---

## 4.4 Módulo de Geração Aumentada por Recuperação (RAG)

Enquanto a ideação inicial explora o conhecimento paramétrico do LLM sobre tópicos de computação, a formalização metodológica exige ancoragem estrita e auditável. O Módulo RAG complementa a arquitetura assegurando que conceitos científicos, escolhas de métodos de validação e normas técnicas estejam fundamentados na literatura consolidada.

### 4.4.1 Curadoria e Construção da Base Canônica
A base de conhecimento foi estruturada a partir de **21 manuais e obras de referência de metodologia científica**, cobrindo tanto a literatura clássica quanto diretrizes especializadas em computação e engenharias:
- **Computação:** Wazlawick (*Metodologia de Pesquisa para Ciência da Computação*), Berndtsson et al. (*Thesis Projects: A Guide for Students in Computer Science and Information Systems*), Dugan Jr. & Polanski (*Writing for Computer Science*).
- **Metodologia Geral:** Lakatos & Marconi (*Fundamentos de Metodologia Científica*), Gil (*Como Elaborar Projetos de Pesquisa*), Prodanov & Freitas (*Metodologia do Trabalho Científico*), Silva & Menezes (*Metodologia da Pesquisa e Elaboração de Dissertação*).
- **Comunicação Científica:** Sampaio, Sabadini & Koller (*Produção Científica: Um Guia Prático*), Traina & Traina Jr. (*Como Fazer Pesquisa Bibliográfica*).

### 4.4.2 Pipeline de Fragmentação (*Chunking*) e Vetorização
Os textos originais (em formatos PDF e TXT) foram processados por um pipeline dedicado em `pipeline/`:
1. **Extração Textual:** Extração sem perdas de camadas textuais via PyMuPDF (`fitz`), com saneamento de cabeçalhos e rodapés repetitivos.
2. **Chunking Semântico:** Divisão em blocos de 1.000 caracteres com sobreposição (*chunk overlap*) de 200 caracteres através do `RecursiveCharacterTextSplitter`. Essa parametrização assegura a integridade das sentenças metodológicas e a preservação do contexto conceitual entre fragmentos contíguos.
3. **Vetorização e Indexação:** Cada fragmento foi convertido em representação vetorial densa de 3.072 dimensões utilizando o modelo `models/gemini-embedding-001` e armazenado no banco vetorial local persistente **ChromaDB**, totalizando **5.839 fragmentos indexados**.

### 4.4.3 Mecanismos de RAG: Diálogo Dinâmico vs. Consultoria Estrita
O artefato opera o RAG sob duas abordagens complementares:

* **1. RAG Dinâmico em Diálogo Livre (`stream_dialogo_academico`):**  
  Ativado após a síntese da estratégia. Executa busca por similaridade de cosseno ($k=3$) para enriquecer o contexto do orientador com fragmentos dos manuais, permitindo esclarecer dúvidas sobre tamanho amostral, métricas de avaliação ou estruturação de experimentos em tom didático.
* **2. Consultoria Metodológica Estrita e Anti-Alucinação (`stream_resposta_rag_estrito`):**  
  Opera em temperatura `0.0` com $k=4$ fragmentos e diretivas de fidelidade estrita. Se o tópico consultado não possuir correspondência nos manuais indexados, o modelo é instruído a declarar a ausência de evidência na base e recomendar a busca de literatura complementar, suprimindo alucinações. As obras consultadas são formatadas e listadas automaticamente segundo os padrões da ABNT.

---

## 4.5 Síntese Automatizada e Compilação do Documento Estratégico

A culminação do processo socrático ocorre quando o discente aciona o botão *"Concluir Coleta e Gerar Documento Estratégico"*. O histórico completo de perguntas e escolhas é submetido à cadeia de síntese (`stream_gerar_documento_estrategico`), que produz um documento de planejamento formal.

### 4.5.1 Estrutura do Documento Estratégico
O documento gerado possui estrutura obrigatória padronizada:
1. **Diagnóstico do Perfil e Delimitação Temática:** Grande área, subárea delimitada, descrição do problema e formulação explícita da **Pergunta Norteadora**.
2. **Caminhos Metodológicos Propostos:** Apresentação comparativa de duas abordagens:
   - *Caminho A (Abordagem Principal):* Proposta principal com objetivos, métodos, fontes de dados e estratégias de mitigação.
   - *Caminho B (Abordagem Alternativa):* Alternativa com foco técnico distinto para mitigar dependência de um único método.
3. **Recomendações e Próximos Passos:** Diretrizes de busca bibliográfica e preparação de reuniões com o orientador humano.
4. **Cronograma Estimado de Execução por Etapas:** Planejamento neutro estruturado em 4 etapas (Etapa 1: Fundamentação; Etapa 2: Coleta de Dados; Etapa 3: Execução Metodológica; Etapa 4: Análise e Redação Final).

### 4.5.2 Motor de Diagramação e Exportação em PDF
Para transformar o texto Markdown em um artefato tangível para apresentação ao professor orientador, foi desenvolvido um compilador tipográfico proprietário em `services/pdf_service.py` baseado na biblioteca `FPDF`:
- **Tratamento de Símbolos e Latin-1:** Sanitização de caracteres Unicode complexos, emojis e fórmulas matemáticas simples em LaTeX.
- **Engine Anti-Quebra e Margem de Segurança:** Implementação de um buffer de segurança de 3,5 mm antes da margem física direita do documento, combinada com algoritmos de reconciliação de palavras partidas (`corrigir_palavras_quebradas`), garantindo que palavras nunca sejam fragmentadas nas quebras de linha e prevenindo caracteres órfãos.

---

## 4.6 Persistência e Rastreabilidade Científica no Firebase

Para atender ao delineamento experimental e viabilizar a auditoria de dados no estudo empírico, o sistema utiliza o **Google Cloud Firestore** como banco de dados NoSQL distribuído.

### 4.6.1 Modelagem de Dados
O esquema foi estruturado em duas coleções aninhadas:

```
coleção: conversas
└── documento: {conversa_id} (ID alfanumérico auto-gerado de 20 chars)
    ├── user_id: "discente@cin.ufpe.br"
    ├── tipo: "estrategia" | "rag_estrito"
    ├── status: "em_andamento" | "concluido"
    ├── title: "[Estratégia] Visão Computacional no Diagnóstico..."
    ├── total_mensagens: 8
    ├── documento_gerado: "# Estratégia de Pesquisa..." (Markdown)
    ├── feedback: { "rating": 5, "comentario": "Excelente afunilamento!" }
    ├── created_at: SERVER_TIMESTAMP
    ├── updated_at: SERVER_TIMESTAMP
    │
    └── subcoleção: mensagens
        └── documento: {mensagem_id}
            ├── role: "assistant" | "user"
            ├── fase: "COLETA" | "GERACAO" | "DIALOGO_ABERTO" | "RAG_ESTRITO"
            ├── rodada: 1, 2, 3...
            ├── content: "Texto completo da intervenção..."
            ├── opcoes_oferecidas: ["Opção 1", "Opção 2", "Opção 3", "Opção 4"]
            ├── opcao_selecionada: "Opção 2"
            ├── fontes_consultadas: ["WAZLAWICK, Raul Sidnei...", "LAKATOS..."]
            └── timestamp: SERVER_TIMESTAMP
```

Essa modelagem permite reconstituir com precisão micro-decisões do estudante ao longo de cada turno, facilitando análises posteriores de mineração de processos cognitivos e identificação de padrões temáticos mais procurados no CIn/UFPE.

---

## 4.7 Resiliência Operacional e Gestão de Cotas de Modelos

A fim de garantir disponibilidade contínua durante as sessões experimentais com os estudantes, foram implementados mecanismos de resiliência em `config.py` e `services/ai_service.py`:
- **Mapeamento de Modelos Legados:** Redirecionamento automático de versões depreciadas da API do Google para a família estável `gemini-3.5-flash-lite`.
- **Tratamento de Exaustão de Cotas (HTTP 429) e Falhas Temporárias:** Implementação de interceptadores (*try-catch* com *fallback*) que reexecutam requisições com degradação suave sem interromper a sessão do usuário.
- **Isolamento de Credenciais:** Suporte duplo ao carregamento de segredos de autenticação (variáveis locais `.env` em desenvolvimento e `st.secrets` no Streamlit Cloud para produção).

---

## 4.8 Considerações Finais do Capítulo

A concepção do Orientador Acadêmico Inteligente materializa uma arquitetura híbrida focada em usabilidade, rigor conceitual e reprodutibilidade. Ao combinar a agilidade da entrevista socrática orientada a decisões com a solidez documental do subsistema RAG sobre 21 manuais de metodologia, o artefato estabelece a infraestrutura técnica necessária para a validação empírica detalhada no capítulo seguinte.
