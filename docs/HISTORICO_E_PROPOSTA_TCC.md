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

## 📄 2. Texto da Proposta de TCC (Formatado em 2 Parágrafos / ~17 Linhas)

> O objetivo do trabalho é implementar um agente conversacional inteligente baseado em Modelos de Linguagem de Larga Escala (LLMs) e Geração Aumentada por Recuperação (RAG) para orientação acadêmica, atuando como uma camada de mediação pedagógica que conduz pesquisadores desde a ideação inicial até a estruturação metodológica de projetos de pesquisa. Diferentemente de assistentes conversacionais generalistas, sujeitos a alucinações conceituais e respostas dispersas, o sistema emprega uma dinâmica socrática de afunilamento progressivo ancorada estritamente em uma base vetorial de manuais canônicos de metodologia científica, garantindo rigor normativo, pertinência teórica e controle de coerência em tempo real.
>
> O foco central do estudo é analisar como a combinação entre recuperação de conhecimento especializado e engenharia de prompts direcionada pode guiar a formulação de problemas de pesquisa e hipóteses de forma consistente e pedagogicamente fundamentada. Para validar a eficácia da abordagem proposta, o agente é avaliado sistematicamente em comparação com o uso de LLMs convencionais sem recuperação contextualizada (baseline), mensurando a precisão conceitual das orientações e a maturidade metodológica dos planos de pesquisa gerados. Os resultados visam demonstrar se o orquestrador socrático com RAG supera abordagens estáticas e genéricas, otimizando o processo preliminar de orientação científica e reduzindo barreiras na etapa de delimitação de temas acadêmicos.

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
│    • Gemini 2.5-flash (Streaming)│   │    • Firestore (users/conversas)│
│    • Questionário Socrático + RAG│   │    • Telemetria & Feedbacks     │
│    • stream_dialogo_academico    │   │    • Metadados Analíticos       │
│    • Consultor RAG Estrito (0.0) │   │  - pdf_service.py:              │
│  - rag_service.py:               │   │    • FPDF2 com formatação Latin │
│    • ChromaDB + gemini-embedding │   │  - pipeline/                    │
│    • 5.839 Chunks Metodológicos  │   │    • PyMuPDF / Chunking         │
└──────────────────────────────────┘   └─────────────────────────────────┘
```

---

## 🔬 4. Como o RAG Opera em Todas as Etapas do Sistema

| Modo / Etapa | Como o RAG atua? | Função no Sistema |
| :--- | :--- | :--- |
| **1. Questionário de Ideação** | Consulta o ChromaDB para recuperar diretrizes metodológicas canônicas (delimitação de problema e hipóteses de Lakatos, Gil, Severino). | Modela as 4 opções sugeridas pelo Gemini com fundamentação científica estrita. |
| **2. Síntese da Estratégia** | Consolida o histórico das escolhas do pesquisador em Markdown aprofundado. | Produz o documento com Diagnóstico, Pergunta Norteadora, Caminhos A e B detalhados, Riscos e Próximos Passos. |
| **3. Diálogo Livre Pós-Estratégia** | `stream_dialogo_academico` realiza busca semântica em tempo real conforme as dúvidas do aluno. | Tira dúvidas sobre execução, ferramentas, cronograma e métodos de pesquisa. |
| **4. Consultor Metodológico Fiel** | Temperatura `0.0`, busca os 4 fragmentos mais similares no ChromaDB. | Respostas estritamente fundamentadas na literatura com citação explícita das obras consultadas. |

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

Para responder à pergunta *"Como provar cientificamente que o agente é melhor do que uma IA comum?"*, foram desenhados dois instrumentos de avaliação:

### 📝 Instrumento A: Questionário de Percepção do Aluno (Modelo TAM - *Technology Acceptance Model*)
Aplicado aos estudantes após o uso do sistema (Escala Likert de 1 a 5 estrelas):
1. **Utilidade Percebida:** *"O agente me ajudou a sair de uma ideia genérica de dados e chegar a um problema específico?"*
2. **Facilidade de Decisão (Método Socrático):** *"As 4 opções oferecidas tornaram o processo de escolha mais claro e estruturado?"*
3. **Qualidade do Documento Final:** *"O plano de pesquisa em PDF gerado é claro e viável para ser apresentado a um orientador?"*
4. **Fidelidade Metodológica:** *"As perguntas do agente pareceram fundamentadas nas regras de pesquisa da disciplina?"*

### 🎓 Instrumento B: Rubrica de Avaliação Conceitual (Avaliação Cega por Professores)
Ficha com 4 critérios (notas de 0 a 10) para avaliar planos gerados pelo Agente vs. IA Comum (Baseline):
* **Critério 1: Formulação do Problema** (O problema em Dados é claro, relevante e possui variáveis mensuráveis?).
* **Critério 2: Coerência da Hipótese** (A hipótese proposta tem relação lógica direta com o problema?).
* **Critério 3: Viabilidade dos Dados** (O método de coleta e o dataset sugerido são realistas para o prazo de um TCC?).
* **Critério 4: Rigor Metodológico** (Ausência de erros conceituais ou alucinações normativas).

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
