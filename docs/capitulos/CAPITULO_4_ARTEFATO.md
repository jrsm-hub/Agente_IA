# 4 O ARTEFATO: ARQUITETURA E IMPLEMENTAÇÃO DO AGENTE INTELIGENTE

Neste capítulo, detalha-se a concepção arquitetural, as decisões de projeto e a implementação computacional do artefato desenvolvido nesta pesquisa. A estruturação deste trabalho apoia-se nos preceitos metodológicos da *Design Science Research* (DSR), especificamente na etapa de projeto e desenvolvimento do artefato (HEVNER et al., 2004; VAISHNAVI; KUECHLER, 2015). O artefato computacional produzido consiste em um sistema de mediação pedagógica inteligente voltado à ideação e delimitação metodológica de projetos científicos de graduação em Ciência da Computação, fundamentado na integração entre modelos de linguagem de grande escala (*Large Language Models* - LLMs), um orquestrador dialógico de viés socrático e um módulo de Geração Aumentada por Recuperação (*Retrieval-Augmented Generation* - RAG) ancorado na literatura de referência em metodologia científica.

## 4.1 Visão Geral e Requisitos do Artefato

A transição entre o aprendizado técnico-operacional e a formulação de uma investigação científica rigorosa constitui um dos principais gargalos enfrentados por estudantes de graduação (WAZLAWICK, 2014). Frequentemente, discentes iniciantes confundem a mera construção de um produto de software com a investigação de um problema científico falseável e testável. Para mitigar esse obstáculo cognitivo, o artefato foi projetado para atuar como um mediador reflexivo, guiando o pesquisador através de turnos estruturados de tomada de decisão.

Seguindo as diretrizes clássicas de Engenharia de Software propostas por Sommerville (2019) e Pressman e Maxim (2016), o levantamento das necessidades do sistema foi sintetizado em requisitos funcionais e não funcionais.

No tocante aos requisitos funcionais, o sistema deve:
1. Conduzir entrevistas socráticas adaptativas em turnos graduais, identificando a área de interesse do estudante e afunilando o escopo a partir de perguntas reflexivas;
2. Oferecer, a cada iteração dialógica, quatro alternativas metodológicas fundamentadas, permitindo concomitantemente a inserção de respostas dissertativas livres pelo usuário;
3. Compilar dinamicamente o histórico de decisões do estudante em um Documento Estratégico formal, contendo contextualização temática, pergunta norteadora investigativa, duas abordagens metodológicas comparativas (Caminho A e Caminho B), análise de riscos com planos de contingência e um cronograma estimado de execução por etapas;
4. Exportar o documento gerado em formato PDF com diagramação e normas de apresentação acadêmica;
5. Disponibilizar um ambiente de diálogo aberto pós-estratégia, amparado por busca vetorial semântica em literatura metodológica, para sanar dúvidas sobre execução prática e delineamento experimental;
6. Oferecer um modo especializado de consultoria metodológica estrita com temperatura zero, capaz de auditar rascunhos textuais e emitir pareceres conceituais com citação explícita dos manuais consultados;
7. Persistir com segurança na nuvem as sessões, mensagens, documentos produzidos e avaliações de utilidade dos participantes.

Quanto aos requisitos não funcionais, estabeleceu-se que o artefato deve:
1. Apresentar baixa latência de resposta, com início de exibição em *streaming* textual inferior a dois segundos para preservar o engajamento cognitivo do usuário;
2. Garantir isolamento conceitual na fase de ideação, evitando que buscas vetoriais prematuras injetem ruído terminológico genérico em domínios específicos de Ciência da Computação;
3. Assegurar conformidade conceitual e mitigação de alucinações (JI et al., 2023) nas respostas metodológicas formais, restringindo afirmações aos fragmentos recuperados da literatura;
4. Ser disponibilizado sob arquitetura web aberta e responsiva, operando sem exigir poder de processamento local ou hardware especializado dos estudantes.

## 4.2 Arquitetura Geral da Solução

Para assegurar manutenibilidade, extensibilidade e desacoplamento de responsabilidades, o sistema foi estruturado segundo o padrão de Arquitetura em Camadas (*Layered Architecture*) (BASS; CLEMENTS; KAZMAN, 2012). A organização divide-se em três macrocamadas lógicas: Camada de Apresentação e Interação, Camada de Cognição e Serviços de Inteligência Artificial, e Camada de Persistência e Recuperação de Dados.

O Quadro 4.1 sintetiza a organização dos módulos, suas responsabilidades sistêmicas e as tecnologias adotadas em cada estrato.

**Quadro 4.1 – Especificação dos Módulos e Tecnologias da Arquitetura em Camadas**

| Camada Arquitetural | Módulo / Arquivo | Responsabilidade Principal | Tecnologias Utilizadas |
| :--- | :--- | :--- | :--- |
| **Apresentação e Interação** | `app.py` | Renderização da interface web, autenticação institucional por matrícula/e-mail, controle de turnos dialógicos, botões dinâmicos de resposta e visualização de histórico. | Python, Streamlit Framework, Streamlit State Management. |
| **Cognição e Orquestração** | `services/ai_service.py` | Engenharia de prompts socráticos, orquestração de streaming, síntese do documento estratégico, diálogo aberto e consultoria anti-alucinação. | LangChain Core, Google Generative AI SDK, PromptTemplates. |
| **Recuperação Vetorial (RAG)** | `services/rag_service.py` | Indexação densa, cálculo de similaridade de cosseno ($k$-NN) e recuperação de fragmentos conceituais dos manuais de metodologia. | ChromaDB, Gemini Embeddings (`gemini-embedding-001`). |
| **Diagramação e Tipografia** | `services/pdf_service.py` | Normalização tipográfica Latin-1, conversão de Markdown e LaTeX, controle de margens e compilação do relatório final em PDF. | FPDF2, RegEx, PyMuPDF. |
| **Persistência de Dados** | `services/firebase_service.py` | Armazenamento de metadados, controle multi-inquilino de sessões, persistência integral dos diálogos e coleta de telemetria empírica. | Google Cloud Firestore, Firebase Admin SDK. |

Fonte: Elaborado pelo autor (2026).

A comunicação entre a camada de apresentação (`app.py`) e os serviços de inteligência é estritamente unidirecional. A interface solicita a geração de turnos dialógicos ou pareceres repassando apenas o histórico conversacional acumulado, enquanto a camada de cognição devolve geradores em fluxo contínuo (*generators* em Python). Esse desacoplamento permite que substituições no modelo de linguagem subjacente ocorram sem a necessidade de alterações no leiaute de usuário.

## 4.3 Módulo Dialógico Socrático para Afunilamento Temático

O afunilamento temático baseia-se no método de questionamento socrático (PAUL; ELDER, 2006), no qual o orientador não fornece respostas prontas ou direcionamentos impositivos, mas formula indagações dialéticas que conduzem o interlocutor à reflexão crítica sobre as fronteiras, viabilidade e pressupostos de sua própria investigação.

### 4.3.1 O Desacoplamento entre Diálogo Socrático e Recuperação Vetorial

Uma premissa de engenharia relevante adotada nesta pesquisa reside no desacoplamento entre a busca vetorial (RAG) e os turnos iniciais da entrevista socrática. Em experimentos preliminares com arquiteturas integradas, observou-se que submeter as respostas do discente a uma busca por similaridade em manuais metodológicos a cada turno introduzia duas fragilidades críticas:

Em primeiro lugar, a sobrecarga de processamento computacional aumentava o tempo de resposta em cerca de 3,5 a 4,8 segundos por turno, prejudicando a fluidez dialógica essencial à manutenção da concentração do aluno. Em segundo lugar, como os fragmentos dos livros de metodologia abordam primordialmente normas genéricas de pesquisa científica (amostragem, hipóteses formais, métodos bibliográficos), sua injeção no contexto de áreas específicas de Ciência da Computação — como compiladores, processamento de sinais ou visão computacional — gerava ruído conceitual, diluindo o vocabulário técnico de ponta necessário à proposição de tópicos contemporâneos de investigação.

Dessa forma, optou-se por restringir o motor do questionário socrático à engenharia de prompts sobre o conhecimento paramétrico do modelo `gemini-3.5-flash-lite`, preservando a agilidade do diálogo (latência média de 1,2 segundo) e resguardando a busca vetorial para as etapas subsequentes de fundamentação procedimental e auditoria formal.

### 4.3.2 Parametrização e Engenharia de Prompts Socráticos

O comportamento do orquestrador dialógico é regido por diretivas formais consolidadas no modelo de prompt executado pela função `stream_gerar_proxima_pergunta`. A instrução define que o agente atue na persona de um orientador experiente, exigindo sobriedade acadêmica, clareza didática e a apresentação estruturada de exatamente quatro rumos metodológicos possíveis.

A calibragem do prompt foi refinada a partir dos feedbacks colhidos nas sessões com discentes do Centro de Informática. Suprimiram-se termos excessivamente rebuscados em favor de uma linguagem didática, direta e encorajadora, preservando o rigor investigativo sem impor barreiras de compreensão vocabular. Cada intervenção do sistema encerra-se com uma pergunta instigante seguida de quatro opções numeradas, as quais são interpretadas pela interface gráfica e exibidas como botões de seleção rápida. O aluno mantém autonomia irrestrita para optar por um dos caminhos oferecidos ou inserir uma consideração dissertativa própria através do campo de texto.

## 4.4 Módulo de Geração Aumentada por Recuperação (RAG)

Embora o conhecimento paramétrico dos LLMs seja vasto, tais modelos são suscetíveis a alucinações conceituais, imprecisões citacionais e anacronismos bibliográficos quando requisitados a emitir pareceres científicos rigorosos (LEWIS et al., 2020; ZHANG et al., 2023). Para assegurar ancoragem teórica inequívoca, implementou-se um subsistema de RAG fundamentado na literatura de metodologia científica.

### 4.4.1 Curadoria da Base de Literatura Metodológica de Referência

A base de conhecimento foi construída a partir da seleção criteriosa de 21 obras especializadas, abrangendo manuais consolidados de metodologia científica e diretrizes focadas especificamente na pesquisa em computação e ciências exatas. Destacam-se as seguintes obras de referência:

1. **Metodologia de Pesquisa em Computação:** obras de Raul Sidnei Wazlawick (2014) (*Metodologia de Pesquisa para Ciência da Computação*) e o manual internacional de Mikael Berndtsson e colaboradores (2008) (*Thesis Projects: A Guide for Students in Computer Science and Information Systems*);
2. **Fundamentos e Métodos de Pesquisa:** manuais consolidados de Eva Maria Lakatos e Marina de Andrade Marconi (2017) (*Fundamentos de Metodologia Científica*), Cleber Prodanov e Ernani Freitas (2013) (*Metodologia do Trabalho Científico*), Antonio Carlos Gil (2017) (*Como Elaborar Projetos de Pesquisa*) e Edna Lúcia da Silva e Estera Muszkat Menezes (2005) (*Metodologia da Pesquisa e Elaboração de Dissertação*);
3. **Escrita e Comunicação Científica:** diretrizes especializadas de Robert Dugan Jr. e Virginia Polanski (2005) (*Writing for Computer Science*), Silvia Koller e colaboradores (2014) (*Manual de Produção Científica*) e Agma Traina e Caetano Traina Jr. (2009) (*Como Fazer Pesquisa Bibliográfica*).

### 4.4.2 Processamento de Texto, Fragmentação e Vetorização

O pipeline de dados, documentado nos módulos do diretório `pipeline/`, executa o tratamento das obras de referência através de três fases sequenciais:

Na primeira fase, os arquivos em formatos PDF e TXT são submetidos à extração textual por meio da biblioteca PyMuPDF, removendo elementos paratextuais repetitivos, como numerações de página descontextualizadas e cabeçalhos tipográficos.

Na segunda fase, procede-se à fragmentação semântica (*chunking*) através do algoritmo `RecursiveCharacterTextSplitter`. Os textos foram segmentados em blocos de 1.000 caracteres com sobreposição (*chunk overlap*) de 200 caracteres. A sobreposição garante que princípios metodológicos contínuos não sofram truncamento nas fronteiras dos fragmentos, preservando a coerência das sentenças citacionais.

Na terceira fase, cada fragmento textual foi convertido em um vetor denso de 3.072 dimensões utilizando o modelo `models/gemini-embedding-001`, sendo subsequentemente indexado no banco vetorial local persistente ChromaDB. O processo resultou em uma base de 5.839 fragmentos indexados, acompanhados de metadados padronizados contendo a autoria e a obra de origem segundo as normas da ABNT.

### 4.4.3 Mecanismos Operacionais: Diálogo Livre vs. Consultoria Estrita

O subsistema de RAG atua sob dois delineamentos operacionais com objetivos epistemológicos distintos:

No **Diálogo Metodológico Pós-Estratégia** (`stream_dialogo_academico`), o RAG opera com busca por similaridade de cosseno ($k=3$), enriquecendo o contexto do modelo para esclarecer dúvidas procedimentais surgidas após a emissão do plano de pesquisa, como a escolha de métricas em aprendizado de máquina ou cuidados na coleta de amostras.

No modo de **Consultoria Metodológica Estrita e Auditoria Textual** (`stream_resposta_rag_estrito`), o agente opera sob temperatura zero ($T=0,0$) e parâmetro de recuperação ampliado ($k=4$). O prompt impõe diretrizes estritas de fidelidade documental: o parecer deve apoiar-se exclusivamente nos trechos recuperados. Caso a base documental não contenha subsídios suficientes para responder à indagação do pesquisador, o modelo é instruído a declarar explicitamente a insuficiência de dados em vez de inferir suposições. Ao final da resposta, o sistema anexa automaticamente a relação formal das obras e autores consultados.

## 4.5 Síntese Automatizada e Compilação Tipográfica em PDF

Ao término dos turnos socráticos de ideação, o histórico consolidado é submetido à rotina de geração documental estruturada (`stream_gerar_documento_estrategico`), culminando na formalização do Documento Estratégico de Pesquisa.

### 4.5.1 Estrutura do Documento Estratégico

Para assegurar completude científica, o documento sintetizado organiza-se em quatro seções formais:
1. **Diagnóstico do Perfil e Delimitação Temática:** contextualiza a grande área e o nicho tecnológico selecionado, sintetizando a lacuna investigativa e formalizando a **Pergunta Norteadora** em formato interrogativo direto;
2. **Caminhos Metodológicos Propostos:** apresenta de forma comparativa duas abordagens de execução:
   - *Caminho A (Abordagem Principal):* detalha a técnica investigativa primária, especificando hipóteses, bases de dados pretendidas (incentivando repositórios abertos caso haja barreiras de acesso) e protocolos de teste;
   - *Caminho B (Abordagem Alternativa):* introduz uma alternativa metodológica ou escopo concorrente, assegurando que o pesquisador disponha de contingência técnica caso enfrente inviabilidade operacional no plano principal;
3. **Recomendações e Próximos Passos:** instrui o estudante sobre termos-chave para revisões bibliográficas e tópicos prioritários para alinhamento com seu professor orientador humano;
4. **Cronograma Estimado de Execução por Etapas:** distribui as metas do trabalho em quatro fases universais e neutras (Etapa 1: Fundamentação Teórica; Etapa 2: Coleta de Dados e Ambiente; Etapa 3: Execução Metodológica e Avaliação; Etapa 4: Análise de Resultados e Redação Final).

### 4.5.2 Motor de Diagramação Tipográfica em PDF

A materialização do Documento Estratégico em um relatório de padrão institucional é viabilizada pelo módulo `services/pdf_service.py`, desenvolvido sobre a biblioteca FPDF. O motor tipográfico implementa rotinas especializadas de conversão:
1. **Sanitização de Caracteres e Expressões Matemáticas:** normalização de caracteres Unicode para o padrão de codificação Latin-1 e conversão automática de notações matemáticas simples em LaTeX para representação alfanumérica limpa;
2. **Prevenção de Quebras Indevidas e Caracteres Órfãos:** implementação de uma margem de segurança de 3,5 mm antes do limite físico lateral da página, combinada com algoritmos heurísticos de reconciliação textual (`corrigir_palavras_quebradas`). Essa solução impede que o motor nativo da biblioteca divida termos no final das linhas, garantindo a integridade ortográfica dos vocábulos e eliminando a ocorrência de letras isoladas nas margens do relatório.

## 4.6 Persistência de Dados e Rastreabilidade Científica

Para viabilizar a auditoria empírica do delineamento intra-sujeito da pesquisa, o artefato utiliza o banco de dados orientado a documentos Google Cloud Firestore. As informações são organizadas na coleção principal denominada `conversas`, vinculada a subcoleções de `mensagens`.

A cada turno dialógico, o sistema persiste os metadados de identificação do discente, o identificador da sessão, o carimbo de data/hora do servidor, a fase do atendimento (`COLETA`, `GERACAO`, `DIALOGO_ABERTO` ou `RAG_ESTRITO`), a opção sugerida que foi selecionada pelo aluno e o teor integral do Documento Estratégico gerado. Essa persistência estruturada garante rastreabilidade temporal irrestrita, permitindo correlacionar as preferências dos participantes com os ganhos de maturidade avaliados na pesquisa empírica.

## 4.7 Resiliência Operacional e Considerações Técnicas

A operação contínua do artefato em ambiente computacional compartilhado demandou a incorporação de rotinas de resiliência e controle de falhas. Em `config.py` e `services/ai_service.py`, implementou-se o mapeamento proativo de modelos da família Gemini, assegurando que requisições endereçadas a versões descontinuadas sejam convertidas automaticamente para instâncias ativas (`gemini-3.5-flash-lite`).

Adicionalmente, o sistema incorpora tratamento de erros de exaustão de cota por limite de requisições por minuto (HTTP 429) e quedas temporárias de rede. Por meio de interceptadores com tentativas escalonadas, o artefato reexecuta chamadas interrompidas de forma transparente para o usuário final. No âmbito da segurança de credenciais, o software adota carregamento dinâmico e agnóstico, lendo variáveis locais de ambiente (`.env`) durante o desenvolvimento e segredos criptografados (`st.secrets`) quando implantado em nuvem.

## 4.8 Síntese do Capítulo

Neste capítulo, foram apresentados a fundamentação arquitetural, o desenho dos módulos e os algoritmos de implementação que compõem o Orientador Acadêmico Inteligente. A união entre a mediação reflexiva socrática e a ancoragem estrita em 21 manuais de referência em metodologia científica viabiliza um artefato robusto, responsivo e adaptado às necessidades pedagógicas de discentes de Ciência da Computação. Os procedimentos empíricos e a avaliação dos dados resultantes do uso prático desta ferramenta são objeto de análise e discussão detalhada no capítulo seguinte.

## Referências Bibliográficas Citadas neste Capítulo

BASS, Len; CLEMENTS, Paul; KAZMAN, Rick. **Software Architecture in Practice**. 3. ed. Boston: Addison-Wesley, 2012.

BERNDTSSON, Mikael et al. **Thesis Projects: A Guide for Students in Computer Science and Information Systems**. 2. ed. London: Springer, 2008.

DUGAN JR., Robert F.; POLANSKI, Virginia G. Writing for Computer Science: A Taxonomy of Writing Tasks and General Advice. **Journal of Computing Sciences in Colleges**, v. 21, n. 2, p. 195-203, 2005.

GIL, Antonio Carlos. **Como Elaborar Projetos de Pesquisa**. 6. ed. São Paulo: Atlas, 2017.

HEVNER, Alan R. et al. Design Science in Information Systems Research. **MIS Quarterly**, v. 28, n. 1, p. 75-105, 2004.

JI, Ziwei et al. Survey of Hallucination in Natural Language Generation. **ACM Computing Surveys**, v. 55, n. 12, p. 1-38, 2023.

KOLLER, Silvia H.; COUTO, Maria Clara P. de; HOHENDORFF, Jean von (Org.). **Manual de Produção Científica**. Porto Alegre: Artmed, 2014.

LAKATOS, Eva Maria; MARCONI, Marina de Andrade. **Fundamentos de Metodologia Científica**. 8. ed. São Paulo: Atlas, 2017.

LEWIS, Patrick et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In: **Advances in Neural Information Processing Systems (NeurIPS 2020)**, v. 33, p. 9459-9474, 2020.

PAUL, Richard; ELDER, Linda. **The Thinker's Guide to the Art of Socratic Questioning**. Dillon Beach: Foundation for Critical Thinking, 2006.

PRESSMAN, Roger S.; MAXIM, Bruce R. **Engenharia de Software: Uma Abordagem Profissional**. 8. ed. Porto Alegre: AMGH, 2016.

PRODANOV, Cleber Cristiano; FREITAS, Ernani Cesar de. **Metodologia do Trabalho Científico: Métodos e Técnicas da Pesquisa e do Trabalho Acadêmico**. 2. ed. Novo Hamburgo: Feevale, 2013.

SILVA, Edna Lúcia da; MENEZES, Estera Muszkat. **Metodologia da Pesquisa e Elaboração de Dissertação**. 4. ed. Florianópolis: UFSC, 2005.

SOMMERVILLE, Ian. **Engenharia de Software**. 10. ed. São Paulo: Pearson Education do Brasil, 2019.

TRAINA, Agma J. M.; TRAINA JR., Caetano. Como Fazer Pesquisa Bibliográfica. **Revista SBC Horizontes**, Porto Alegre: Sociedade Brasileira de Computação, 2009.

VAISHNAVI, Vijay K.; KUECHLER, William. **Design Science Research Methods and Patterns: Innovating Information and Communication Technology**. 2. ed. Boca Raton: CRC Press, 2015.

WAZLAWICK, Raul Sidnei. **Metodologia de Pesquisa para Ciência da Computação**. 2. ed. Rio de Janeiro: Elsevier / Campus, 2014.

ZHANG, Yue et al. Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models. **arXiv preprint arXiv:2309.01219**, 2023.
