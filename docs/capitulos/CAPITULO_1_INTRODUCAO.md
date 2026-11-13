# 1 INTRODUÇÃO

A formação universitária em Ciência da Computação demanda o desenvolvimento não apenas de proficiência técnica no projeto e na implementação de sistemas de software, mas também da capacidade de conduzir investigações científicas estruturadas e metodologicamente rigorosas (WAZLAWICK, 2014). No contexto dos cursos de graduação, essa exigência materializa-se primordialmente na elaboração do Trabalho de Conclusão de Curso (TCC), momento em que o discente deve demonstrar autonomia na identificação de lacunas de conhecimento, na formulação de hipóteses falseáveis e na seleção de procedimentos empíricos de validação (BERNDTSSON et al., 2008).

Contudo, a fase inicial desse processo investigativo — correspondente à ideação temática e à definição do problema de pesquisa — representa historicamente uma das etapas de maior atrito cognitivo e emocional para os estudantes. Frequentemente imersos em disciplinas práticas de desenvolvimento tecnológico, os alunos enfrentam dificuldades para delimitar escopos e tendem a confundir o desenvolvimento meramente operacional de uma aplicação de software com a formulação de um problema científico genuíno. Paralelamente, o corpo docente enfrenta limitações severas de tempo para conduzir atendimentos preliminares individualizados de sondagem temática, dada a sobrecarga de orientações e compromissos departamentais.

Diante desse cenário, os avanços recentes no processamento de linguagem natural, impulsionados pelos Modelos de Linguagem de Grande Escala (*Large Language Models* - LLMs) e pelas arquiteturas de agentes inteligentes (VASWANI et al., 2017; BROWN et al., 2020), abrem novas perspectivas para a mediação pedagógica automatizada. Todavia, a simples adoção de chatbots conversacionais genéricos revela-se insuficiente para a orientação científica preliminar, em virtude da propensão desses modelos a gerar informações imprecisas ou fabricadas — fenômeno conhecido como alucinação conceitual (JI et al., 2023) — e da ausência de uma pedagogia dialógica estruturada para o afunilamento de ideias.

Este trabalho propõe, desenvolve e avalia um agente inteligente híbrido voltado à mediação pedagógica na orientação preliminar de pesquisas científicas em Ciência da Computação. Apoiando-se nos preceitos da *Design Science Research* (DSR) (HEVNER et al., 2004; VAISHNAVI; KUECHLER, 2015), o artefato computacional integra um orquestrador dialógico socrático voltado ao afunilamento progressivo de tópicos, complementado por um módulo de Geração Aumentada por Recuperação (*Retrieval-Augmented Generation* - RAG) (LEWIS et al., 2020) ancorado na literatura de referência em metodologia científica, culminando na síntese automatizada de um documento estratégico de planejamento investigativo.

## 1.1 Contextualização e Motivação

O Centro de Informática da Universidade Federal de Pernambuco (CIn/UFPE) destaca-se nacional e internacionalmente pela formação técnica de excelência e pela expressiva produção científica em áreas como Inteligência Artificial, Engenharia de Software, Ciência de Dados e Sistemas de Computação. No entanto, o percurso acadêmico dos discentes que ingressam nas etapas finais de graduação reflete uma assimetria recorrente: enquanto as habilidades de codificação e resolução de problemas técnicos são amplamente consolidadas ao longo do curso, as competências voltadas à escrita e ao método científico costumam ser desenvolvidas de forma tardia (DUGAN JR.; POLANSKI, 2005).

A fase de ideação é marcada, em sua origem, pelo fenômeno da "página em branco". Ao serem solicitados a definir um tema para seu TCC, muitos estudantes manifestam apenas interesses difusos por grandes áreas em evidência no mercado, como "aplicar inteligência artificial" ou "otimizar bancos de dados", sem discernimento sobre quais problemas de pesquisa permanecem em aberto na literatura. Essa amplitude excessiva gera ansiedade, protelação e retrabalho, resultando frequentemente em propostas desprovidas de recorte metodológico claro ou fundamentação empírica viável.

Por outro lado, o processo de afunilamento temático exige tradicionalmente múltiplas reuniões de orientação preliminar, nas quais o professor orientador atua questionando as premissas do aluno, apontando inviabilidades de escopo e sugerindo caminhos metodológicos alternativos. Dado o elevado contingente de concluintes por semestre, esse modelo puramente presencial sobrecarrega os orientadores com turnos repetitivos de nivelamento básico, restringindo o tempo disponível para discussões aprofundadas sobre o núcleo das pesquisas.

A motivação central desta pesquisa reside, portanto, na oportunidade de conceber um artefato tecnológico capaz de atuar como mediador preliminar nesse intervalo crítico. Ao oferecer um suporte dialógico interativo e pedagogicamente fundamentado, busca-se empoderar o discente na transição de uma ideia preliminar para uma proposta estruturada, otimizando o diálogo posterior entre o aluno e seu professor orientador humano.

## 1.2 Definição do Problema de Pesquisa

A problemática central investigada neste trabalho situa-se na interseção entre a mediação pedagógica da metodologia científica e a engenharia de sistemas baseados em inteligência artificial generativa. Ela pode ser caracterizada por duas dimensões complementares: a dimensão pedagógica discente e a dimensão técnico-computacional dos modelos de linguagem.

Sob a perspectiva pedagógica discente, identifica-se a carência de instrumentos interativos que promovam a reflexão crítica gradual. Ambientes de orientação passivos (como apostilas estáticas ou repositórios de TCCs anteriores) não oferecem resposta adaptativa às incertezas do estudante. Já o uso não orientado de ferramentas comerciais de inteligência artificial tende a fornecer respostas prontas e superficiais, suprimindo o esforço reflexivo necessário para que o aluno compreenda as razões teóricas subjacentes às suas escolhas metodológicas.

Sob a perspectiva técnico-computacional, os LLMs de propósito geral não possuem garantias intrínsecas de fidelidade normativa. Quando instados a estruturar metodologias de pesquisa, esses modelos frequentemente sugerem procedimentos genéricos, citam referências bibliográficas inexistentes ou negligenciam normas técnicas e cuidados rigorosos de amostragem e falseabilidade (ZHANG et al., 2023).

Diante dessa lacuna, formula-se a seguinte pergunta norteadora de pesquisa:

> *Em que medida uma arquitetura computacional híbrida, que integra a orquestração dialógica socrática com a recuperação aumentada por geração (RAG) ancorada na literatura de metodologia científica, é capaz de mitigar a sobrecarga cognitiva na ideação temática e promover o ganho de maturidade metodológica em estudantes de graduação em Ciência da Computação?*

## 1.3 Objetivos

Para responder à pergunta norteadora e orientar o ciclo de desenvolvimento do artefato, estabeleceram-se o objetivo geral e os objetivos específicos a seguir.

### 1.3.1 Objetivo Geral

Projetar, implementar e avaliar um agente conversacional inteligente de arquitetura híbrida — integrando um orquestrador dialógico socrático e um módulo de Geração Aumentada por Recuperação (RAG) ancorado em literatura metodológica de referência — destinado à mediação pedagógica na ideação, delimitação de temas e estruturação metodológica preliminar de pesquisas acadêmicas em Ciência da Computação.

### 1.3.2 Objetivos Específicos

1. **Conceber e calibrar o orquestrador dialógico socrático:** projetar fluxos conversacionais estruturados em turnos de decisão progressiva, oferecendo conjuntos balanceados de quatro alternativas metodológicas por rodada para mitigar o atrito cognitivo da ideação inicial;
2. **Construir e indexar a base vetorial de metodologia científica:** selecionar e processar um acervo de 21 obras de referência em metodologia científica e pesquisa em computação, estabelecendo um pipeline de extração, fragmentação semântica e vetorização densa em banco vetorial dedicado;
3. **Desenvolver o artefato computacional integrado:** implementar a arquitetura de software em Python com interface interativa web, contemplando síntese automatizada de Documento Estratégico em Markdown, motor de compilação tipográfica em PDF e persistência na nuvem de dados e telemetria;
4. **Avaliar empiricamente a eficácia e a aceitação do artefato:** conduzir um estudo experimental intra-sujeito com estudantes de graduação do Centro de Informática da UFPE, mensurando o ganho de maturidade conceitual entre o pré-teste e o pós-teste através de rubrica analítica multicritério e avaliando a utilidade e facilidade de uso percebidas por meio do Modelo de Aceitação Tecnológica (TAM).

## 1.4 Relevância e Contribuições do Trabalho

A relevância desta pesquisa manifesta-se tanto no domínio da Engenharia de Software e Inteligência Artificial quanto no aprimoramento dos processos de ensino-aprendizagem de metodologia científica no ensino superior:

Em primeiro lugar, o trabalho contribui para a literatura de sistemas baseados em conhecimento ao propor uma solução concreta para o desacoplamento cognitivo entre a ideação criativa e a ancoragem normativa factual. Demonstra-se como conciliar a agilidade de resposta de modelos de linguagem leves com a precisão determinística de bases vetoriais de literatura técnica, prevenindo alucinações sem incorrer em latências computacionais excessivas.

Em segundo lugar, disponibiliza-se para a comunidade acadêmica do Centro de Informática da UFPE um artefato funcional, acessível via navegador web e integralmente implementado sob tecnologias de código aberto, que pode ser incorporado à dinâmica pedagógica das disciplinas de metodologia científica e aos laboratórios de pesquisa.

Em terceiro lugar, a pesquisa fornece evidências empíricas originais sobre a percepção de utilidade e facilidade de uso (TAM) de agentes conversacionais no apoio à escrita acadêmica preliminar, estabelecendo um protocolo experimental replicável para investigações futuras sobre autonomia discente mediada por inteligência artificial.

## 1.5 Organização da Monografia

O presente trabalho está estruturado em seis capítulos, dispostos conforme o fluxo do ciclo de vida da pesquisa:

* O **Capítulo 2 (Fundamentação Teórica e Trabalhos Relacionados)** aborda os fundamentos conceituais de Modelos de Linguagem de Grande Escala, alucinações, arquiteturas RAG, método de questionamento socrático e o Modelo de Aceitação Tecnológica (TAM), além de analisar criticamente o estado da arte e trabalhos correlatos de chatbots acadêmicos;
* O **Capítulo 3 (Metodologia da Pesquisa)** detalha o enquadramento metodológico sob a perspectiva da *Design Science Research* (DSR), o delineamento experimental intra-sujeito adotado, as variáveis controladas e os instrumentos de coleta de dados (rubrica analítica multicritério e questionário TAM);
* O **Capítulo 4 (O Artefato: Arquitetura e Implementação)** descreve a concepção de engenharia do sistema, a especificação dos requisitos funcionais e não funcionais, a arquitetura em camadas, a engenharia de prompts, a estruturação da base ChromaDB, o motor de geração de relatórios em PDF e a persistência no Google Cloud Firestore;
* O **Capítulo 5 (Avaliação Empírica e Análise de Resultados)** apresenta e discute os dados quantitativos e qualitativos colhidos nas sessões experimentais com estudantes do CIn/UFPE, detalhando as métricas de aceitação tecnológica, a mensuração do ganho de maturidade metodológica e a análise de feedbacks;
* O **Capítulo 6 (Conclusão e Trabalhos Futuros)** sintetiza as principais conclusões e contribuições alcançadas pelo trabalho, examina as limitações do estudo e aponta direcionamentos para desdobramentos e pesquisas futuras.

## Referências Bibliográficas Citadas neste Capítulo

BERNDTSSON, Mikael et al. **Thesis Projects: A Guide for Students in Computer Science and Information Systems**. 2. ed. London: Springer, 2008.

BROWN, Tom B. et al. Language Models are Few-Shot Learners. In: **Advances in Neural Information Processing Systems (NeurIPS 2020)**, v. 33, p. 1877-1901, 2020.

DUGAN JR., Robert F.; POLANSKI, Virginia G. Writing for Computer Science: A Taxonomy of Writing Tasks and General Advice. **Journal of Computing Sciences in Colleges**, v. 21, n. 2, p. 195-203, 2005.

HEVNER, Alan R. et al. Design Science in Information Systems Research. **MIS Quarterly**, v. 28, n. 1, p. 75-105, 2004.

JI, Ziwei et al. Survey of Hallucination in Natural Language Generation. **ACM Computing Surveys**, v. 55, n. 12, p. 1-38, 2023.

LEWIS, Patrick et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In: **Advances in Neural Information Processing Systems (NeurIPS 2020)**, v. 33, p. 9459-9474, 2020.

VAISHNAVI, Vijay K.; KUECHLER, William. **Design Science Research Methods and Patterns: Innovating Information and Communication Technology**. 2. ed. Boca Raton: CRC Press, 2015.

VASWANI, Ashish et al. Attention Is All You Need. In: **Advances in Neural Information Processing Systems (NeurIPS 2017)**, v. 30, p. 5998-6008, 2017.

WAZLAWICK, Raul Sidnei. **Metodologia de Pesquisa para Ciência da Computação**. 2. ed. Rio de Janeiro: Elsevier / Campus, 2014.

ZHANG, Yue et al. Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models. **arXiv preprint arXiv:2309.01219**, 2023.
