# 🧭 Registro de Histórico do Projeto, Arquitetura & Proposta de TCC
**Projeto:** Agente de Inteligência Artificial Orientador & Estrategista Acadêmico  
**Curso:** Bacharelado em Ciência da Computação (CIn - Centro de Informática)  
**Repositório:** `jrsm-hub/Agente_IA`  
**Data de Consolidação:** 09 de Outubro de 2026  

---

## 📌 1. Títulos Propostos para o TCC

### ⭐ **Opção Principal (Recomendada):**
> **"Desenvolvimento e Avaliação de um Agente Conversacional Socrático com RAG para Orientação e Estruturação de Pesquisas Acadêmicas"**

### Alternativas de Destaque por Linha de Ênfase:
* **Foco em IA & Mitigação de Alucinações:**  
  *"Orientador Virtual Baseado em LLMs e RAG: Mediação Pedagógica e Mitigação de Alucinações na Formulação de Projetos Científicos"*
* **Foco em Afunilamento Temático & Metodologia em Ciência de Dados:**  
  *"Agente Inteligente Baseado em Modelos de Linguagem e Geração Aumentada por Recuperação (RAG) para Afunilamento Temático em Ciência de Dados"*
* **Foco em Engenharia de Software & Design Science Research:**  
  *"Arquitetura e Implementação de um Sistema de Orientação Científica Automatizada Baseado em Geração Aumentada por Recuperação"*

---

## 📄 2. Texto da Proposta de TCC (Formatado em 3 Parágrafos Rigorosos)

> Este trabalho propõe o desenvolvimento e a avaliação empírica de um agente inteligente para mediação pedagógica na orientação preliminar de pesquisas científicas em Ciência da Computação. Apoiando-se nos preceitos da *Design Science Research* (DSR), o artefato computacional implementa uma arquitetura híbrida: integra um orquestrador dialógico socrático voltado ao afunilamento progressivo de tópicos e à formulação de problemas investigativos, complementado por um módulo de Geração Aumentada por Recuperação (RAG) ancorado em manuais canônicos de metodologia científica para assegurar conformidade normativa, rigor conceitual e esclarecimento técnico procedimental.
>
> A investigação tem como objetivo caracterizar e mitigar as principais barreiras enfrentadas por estudantes de graduação na fase de ideação, frequentemente marcada por temas excessivamente amplos e pela confusão entre o desenvolvimento meramente operacional de software e a formulação de problemas científicos testáveis. O sistema guia o pesquisador através de turnos estruturados de tomada de decisão com opções metodológicas fundamentadas, culminando na síntese automatizada de um documento estratégico de pesquisa composto por delimitação contextual, pergunta norteadora, abordagens metodológicas comparativas e mitigação de riscos de execução.
>
> A validação do artefato será conduzida mediante estudo piloto com estudantes do Centro de Informática da Universidade Federal de Pernambuco (CIn/UFPE). O protocolo adota um delineamento experimental intra-sujeito, mensurando o ganho de maturidade metodológica através de uma rubrica analítica multicritério aplicada antes e após o uso da ferramenta, enquanto questionários fundamentados no modelo de aceitação tecnológica (*Technology Acceptance Model* - TAM) controlam variáveis intervenientes de formação prévia e avaliam a utilidade e facilidade percebidas. Com isso, busca-se demonstrar a viabilidade de agentes inteligentes como instrumentos viáveis de apoio à autonomia acadêmica discente.

---

## 🏛️ 3. Arquitetura Técnica do Sistema

O sistema é construído em Python, seguindo uma arquitetura limpa e modular desacoplada:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   STREAMLIT UI (app.py)                                │
│   - Autenticação por Matrícula/E-mail (Multi-usuário)                  │
│   - Interface de Chat com Streaming Nativo (st.write_stream)           │
│   - Botões Dinâmicos de Respostas Rápidas (1 a 4)                      │
│   - Histórico Completo de Interações e Perguntas Socráticas Visíveis   │
│   - Compilação & Download de Documento Estratégico em PDF (FPDF2)      │
│   - Coleta de Feedback Likert (1 a 5 estrelas) + Comentários           │
└───────────────────┬───────────────────────────────┬────────────────────┘
                    │                               │
┌───────────────────▼──────────────┐   ┌────────────▼────────────────────┐
│      SERVICES / AI & RAG         │   │      PERSISTÊNCIA & DADOS       │
│  - ai_service.py:                │   │  - firebase_service.py:         │
│    • Gemini Flash Lite Streaming │   │    • Firestore (users/conversas)│
│    • Orquestrador Socrático      │   │    • Telemetria & Feedbacks     │
│    • stream_dialogo_academico    │   │    • Metadados Analíticos       │
│    • Consultor RAG Estrito (0.0) │   │  - pdf_service.py:              │
│  - rag_service.py:               │   │    • FPDF2 com formatação Latin │
│    • ChromaDB + gemini-embedding │   │  - pipeline/                    │
│    • 5.839 Chunks Metodológicos  │   │    • PyMuPDF / Chunking         │
└──────────────────────────────────┘   └─────────────────────────────────┘
```

---

## 🔬 4. Como o RAG e o Diálogo Socrático Operam no Sistema (Arquitetura Híbrida)

Para garantir máxima responsividade (~1.5s) e evitar injeção de ruído conceitual durante a ideação, o sistema distribui a cognição entre dois pilares complementares:

| Módulo / Fase | Mecanismo de IA | Função e Justificativa Científica |
| :--- | :--- | :--- |
| **1. Questionário de Ideação (Socrático)** | Engenharia de Prompts Direcionada (`gemini-flash-lite-latest`) | Guia o afunilamento em árvore de decisão temática e gera 4 opções metodológicas concretas. Não realiza busca vetorial em cada clique para evitar latência e garantir que os tópicos de domínio (IA, Redes, Dados) fluam sem contaminação por textos metodológicos genéricos. |
| **2. Síntese do Documento Estratégico** | Cadeia de Sumarização Estruturada | Compila o histórico das escolhas em Diagnóstico, Pergunta Norteadora, Caminhos A e B comparativos, Riscos e Próximos Passos. |
| **3. Diálogo Livre Pós-Estratégia** | RAG Dinâmico (`stream_dialogo_academico`) | Consulta os 5.839 chunks do ChromaDB para sanar dúvidas do estudante sobre como conduzir sua pesquisa, escolher amostras e validar hipóteses. |
| **4. Consultoria Metodológica & Auditoria** | RAG Estrito (`stream_resposta_rag_estrito`) | Análise formal de rascunhos em PDF/TXT e consultas metodológicas estritas (temperatura 0.0), com verificação anti-alucinação e citação explícita das obras consultadas. |

---

## 🗄️ 5. Esquema de Dados no Firebase Firestore

```
coleção: conversas
└── documento: {conversa_id}
    ├── user_id: "aluno@universidade.br"
    ├── tipo: "estrategia" | "rag_estrito"
    ├── status: "em_andamento" | "concluido"
    ├── title: "🧭 Pesquisa em Ciência de Dados"
    ├── total_mensagens: 8
    ├── documento_gerado: "# Estratégia de Pesquisa..." (Markdown)
    ├── feedback: { "rating": 5, "comentario": "Excelente afunilamento!" }
    ├── created_at: SERVER_TIMESTAMP
    ├── updated_at: SERVER_TIMESTAMP
    │
    └── subcoleção: mensagens
        └── documento: {msg_id}
            ├── role: "assistant" | "user"
            ├── fase: "COLETA" | "GERACAO" | "DIALOGO_ABERTO" | "RAG_ESTRITO"
            ├── rodada: 1, 2, 3...
            ├── content: "..."
            ├── opcoes_oferecidas: ["Opção 1", "Opção 2", "Opção 3", "Opção 4"]
            ├── opcao_selecionada: "Opção 1"
            ├── fontes_consultadas: ["Lakatos & Marconi...", "Gil, A. C..."]
            └── timestamp: SERVER_TIMESTAMP
```

---

## ⚖️ 6. Comparativo com Trabalho Correlato: Franciesca Goulart Santos (FURG, 2025)

* **Trabalho Analisado:** *Criação de um Chatbot para o Repositório Institucional da FURG* (Biblioteconomia, 2025).
* **Diferenciais Científicos e Técnicos do Nosso Trabalho (Ciência da Computação):**
  1. **Engenharia de Software Proprietária:** Desenvolvimento de um software full-stack aberto em Python (Streamlit, LangChain, ChromaDB, Firebase), enquanto o trabalho anterior configurou uma ferramenta No-Code proprietária (Botpress Cloud).
  2. **Pipeline RAG Customizado e Auditável:** Indexação customizada de 5.839 fragmentos com `gemini-embedding-001` e ChromaDB local, permitindo controle de similaridade de cosseno, k-neighbors e chunking semântico.
  3. **Cognição Socrática Ativa vs. FAQ Reativo:** Nosso agente é proativo (conduz entrevistas, afunila temas e formula hipóteses científicas), enquanto o bot da FURG atua como FAQ reativo para buscas documentais.
  4. **Persistência Multi-inquilino e Geração de Artefatos:** Autenticação por usuário no Firestore e compilação dinâmica de relatórios em PDF com formatação tipográfica acadêmica.

---

## 🛠️ 7. Histórico de Problemas Encontrados e Resoluções Técnicas

| Erro / Problema | Causa Raiz | Solução Aplicada |
| :--- | :--- | :--- |
| **Depreciação de Embeddings Gemini** | `models/text-embedding-004` retornou erro 404/depreciação na API Gemini v1beta. | Migração para `models/gemini-embedding-001` (3072 dimensões) e reindexação dos 5.839 fragmentos. |
| **`ImportError` no Streamlit Cloud** | Tentativa de importar `from langchain.agents import Tool` (removido no LangChain 0.3+). | Substituição por `from langchain_core.tools import Tool` e remoção de dependências legadas. |
| **`ImportError: AgentExecutor`** | `AgentExecutor` e dependências do LangChain Hub depreciadas/incompatíveis no ambiente de nuvem Linux. | Criação de agente nativo e padronização funcional `stream_dialogo_academico` em `ai_service.py`. |
| **`AttributeError: 'bytearray' object has no attribute 'encode'`** | `fpdf2.output()` já retorna `bytes`/`bytearray`, diferente do `fpdf` antigo que retornava string. | Adição de verificação de tipo dinâmica com `isinstance(out, (bytes, bytearray))` em `pdf_service.py`. |
| **`NameError: StreamlitCallbackHandler`** | Instanciação de callback legado que foi removido das importações. | Substituição completa por streaming nativo do Streamlit (`st.write_stream`). |
| **`AttributeError: 'DialogoAcademicoAgent' object has no attribute 'stream'`** | Objeto em cache no `st.session_state` de sessão anterior aberta no navegador. | Refatoração para chamada funcional direta `stream_dialogo_academico` sem depender de instâncias em cache. |

---

## 🎯 8. Alinhamentos com a Orientadora (CIn / Turma de Metodologia / Área de Dados)

Durante a reunião de orientação, foram definidos os seguintes eixos centrais de atuação:

### 1. Afunilamento em Ciência de Dados no CIn:
Conectar as opções socráticas do agente com as 4 grandes linhas de pesquisa em Dados do CIn:
* **Aprendizado de Máquina & Modelagem Preditiva** (Classificação, Regressão, Detecção de Anomalias).
* **Processamento de Linguagem Natural (PLN / LLMs)** (Análise de Sentimentos, RAG, Extração de Informação).
* **Visão Computacional & Processamento de Imagens** (Segmentação, Classificação, Reconhecimento de Padrões).
* **Engenharia de Dados & Big Data** (Pipelines ETL, Governança, Qualidade de Dados).

### 2. Protocolo do Experimento com a Turma de Metodologia:
* **Público-alvo:** Alunos matriculados na disciplina de Metodologia Científica do CIn.
* **Dinâmica:** Os alunos utilizam o Agente de IA para delimitar suas propostas de TCC.
* **Coleta de Dados:** O Firebase grava automaticamente todas as escolhas, o documento final gerado e a avaliação pós-uso.

---

## 📊 9. Instrumentos de Avaliação Científica para a Monografia (Capítulo 5)

Para atender ao rigor exigido pelas bancas e pelos manuais canônicos de pesquisa em computação (Wazlawick, Gil, Marconi & Lakatos), o estudo adota um delineamento experimental **intra-sujeito** com medição de ganho ($\Delta = \text{Pós} - \text{Pré}$) e controle de variáveis intervenientes:

### 🛡️ Controle de Variáveis Intervenientes (Ficha Preliminar de Perfil)
Antes da interação, o participante preenche um cadastro rápido para controle estatístico:
1. **Subárea de Atuação no CIn:** (ex: Inteligência Artificial, Engenharia de Software, Sistemas Embarcados, Redes/Segurança, Teoria).
2. **Experiência Prévia com Pesquisa Científica:** (Nenhuma; Iniciação Científica - PIBIC/PIBITI; Artigo publicado; Trabalho de Conclusão anterior).
3. **Ideia Inicial Pré-Teste (2 a 3 linhas):** Registro sumário do que o aluno imagina pesquisar, sem qualquer auxílio da ferramenta.

*Justificativa Metodológica:* O controle garante demonstrar que o ganho de maturidade nos planos de pesquisa decorre do uso do agente mediador e não de discrepâncias prévias na formação ou facilidade individual dos participantes.

---

### 📝 Instrumento A: Questionário de Aceitação Tecnológica (Modelo TAM - Davis, 1989)
Aplicado logo após o aluno concluir a sessão e exportar o PDF (Escala Likert de 1 a 5: *1 = Discordo Totalmente*, *5 = Concordo Totalmente*):

#### Dimensão 1: Utilidade Percebida (*Perceived Usefulness - PU*)
* **PU1:** O uso do agente me ajudou a transformar uma ideia genérica em um problema de pesquisa bem delimitado e compreensível.
* **PU2:** O documento estratégico gerado em PDF forneceu uma base consistente e estruturada para apresentar ao meu orientador acadêmico.
* **PU3:** As opções metodológicas sugeridas permitiram enxergar caminhos viáveis que eu não havia considerado inicialmente.
* **PU4:** A mediação do agente reduziu o tempo e a ansiedade que eu normalmente teria na fase inicial de definição de tema.

#### Dimensão 2: Facilidade de Uso Percebida (*Perceived Ease of Use - PEOU*)
* **PEOU1:** O formato socrático de perguntas graduais com 4 opções de resposta foi intuitivo e simples de navegar.
* **PEOU2:** A interface do sistema não exigiu esforço mental excessivo ou conhecimento prévio sobre ferramentas de IA.
* **PEOU3:** A transição entre as escolhas rápidas (botões) e a possibilidade de digitar respostas personalizadas foi fluida.
* **PEOU4:** Ficou claro e compreensível de que maneira cada escolha feita ao longo do diálogo moldou o documento final.

---

### 🎓 Instrumento B: Rubrica Analítica Multicritério (Avaliação de Maturidade Metodológica)
Instrumento cego aplicado por avaliadores para mensurar o salto de qualidade entre a ideia inicial do aluno (Pré-teste) e o Documento Estratégico gerado (Pós-teste). Cada critério utiliza escala Likert inequívoca de 1 a 5:

| Critério Analítico | Nível 1 (Incipiente) | Nível 3 (Intermediário) | Nível 5 (Rigoroso / Exequível) |
| :--- | :--- | :--- | :--- |
| **C1: Delimitação do Problema** | Tema genérico sem contexto; ausência de fronteira clara de investigação. | Tema delimitado, mas a pergunta de pesquisa ainda é ampla ou confusa. | Problema de pesquisa claro, específico, contextualizado e formalizado como pergunta investigativa. |
| **C2: Testabilidade da Hipótese** | Mera lista de requisitos de software ou ausência de hipótese científica. | Hipótese esboçada, mas com dificuldade de refutação ou mensuração direta. | Hipótese formal, falseável, diretamente conectada às variáveis do problema de pesquisa. |
| **C3: Viabilidade e Procedimentos** | Não cita métodos, ferramentas ou fontes de dados para execução. | Cita ferramentas e datasets gerais, mas sem detalhar o pipeline ou protocolo. | Delineamento metodológico completo: procedimentos técnicos, fontes de dados, métricas e etapas de validação. |
| **C4: Mapeamento de Riscos** | Não antecipa limitações técnicas ou riscos de inviabilidade. | Menciona riscos genéricos sem apontar planos de mitigação concretos. | Identificação lúcida dos principais gargalos de execução (ex: acesso a dados, latência, viés) e estratégias de contingência. |

$$\Delta_{\text{Maturidade}} = \text{Escore Pós} - \text{Escore Pré}$$

---

## 📈 10. Telemetria e Mineração de Padrões no Firestore

A partir dos dados armazenados no Firebase, a pesquisa analisa:

1. **O que mais aparece nas respostas (Padrões dos Alunos):**
   * Preferência massiva por áreas em alta (ex: LLMs e Visão Computacional vs. Engenharia de Dados clássica).
   * Dificuldade inicial em delimitar escopo na Rodada 1.
2. **Quais temas são mais críticos (Principais Gargalos Metodológicos):**
   * **Definição de Datasets:** O aluno tem a ideia, mas não sabe onde obter a base de dados pública.
   * **Métricas de Avaliação:** Dificuldade em definir métricas formais (F1-score, acurácia, latência, AUC).
   * **Hipóteses Testáveis:** Diferenciação entre "objetivo de desenvolvimento de software" vs. "hipótese de pesquisa científica".

---

## 🧠 11. Comparativo: LLMs em Nuvem vs. SLMs Locais (Small Language Models)

* **Para a Execução do TCC:** O uso de LLMs em nuvem (**Gemini 2.5 Flash / Groq Llama 3.3**) é ideal pela agilidade de deploy no Streamlit Cloud, velocidade de resposta (1 a 3s) e excelente redação em português sem exigir GPU dos usuários.
* **Para a Monografia:** A arquitetura do sistema é posicionada como **agnóstica de modelo** (*Model-Agnostic*), demonstrando que o orquestrador em `ai_service.py` permite substituir a API por **SLMs locais (Llama 3.2 3B, Phi-4, Mistral 7B)** para ambientes que exijam 100% de privacidade e operação offline.

---

## 💻 12. Guia de Setup Rápido para Rodar em Outro Computador

Para clonar e rodar o projeto em um novo computador:

```bash
# 1. Clonar o repositório
git clone https://github.com/jrsm-hub/Agente_IA.git
cd Agente_IA

# 2. Criar e ativar ambiente virtual (Python 3.10+)
python -m venv venv
# No Windows:
.\venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Verificar se o arquivo .env e firebase-credentials.json estão na raiz

# 5. Executar a aplicação Streamlit
streamlit run app.py
```
