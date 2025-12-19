# Projeto de Análise Arquitetural – Governança de Software
**Microsoft JARVIS (HuggingGPT) – Estratégias de Release e Fluxo de Trabalho**

* **Instituição:** Universidade Federal de Sergipe – Departamento de Computação
* **Disciplina:** Engenharia de Software II
* **Professor:** Glauco de Figueiredo Carneiro
* **Atividade:** II – Padrões Arquiteturais de Software
* **Data:** 19/12/2025

---

# 1. Sobre o Projeto

Este repositório contém a análise completa de governança de software realizada pelo grupo sobre o projeto **Microsoft JARVIS (HuggingGPT)**, com foco em:

* Estratégia de Releases
* Modelo de Fluxo de Trabalho (Branching Model)
* Análise manual do histórico do repositório
* Análise automatizada com Modelos de Linguagem (LLMs)
* Comparação crítica entre abordagens humanas e automatizadas

**Objetivo:**
Identificar e avaliar padrões de governança de software, confrontando análise manual especializada com inferências feitas por modelos de linguagem.

---

# 2. Integrantes e Organização

### Integrantes do Grupo
* Arthur Costa Oliveira (202300027104)
* Davi Lira Santana (202300083319)
* Gabriel Batista Barbosa (202300027249)
* João Henrique Britto Bomfim (202300027409)
* Luan Almeida Valença (202300027866)
* Matheus Nascimento dos Santos (202300083810)
* Paulo Henrique Melo Rugani de Sousa (202300027919)
* Tassio Mateus de Carvalho (202300083963)

### Links Importantes
* **Vídeo:** (inserir link)
* **Apresentação:** [Google Slides](https://docs.google.com/presentation/d/1lciOsqsd8QxprPZqVr1Z0ujgT0v0i9GaW81e9FsLrak/edit?usp=sharing)

### Organização do Trabalho
O grupo foi dividido em 4 duplas, cada uma responsável por uma vertente da análise:

* **Dupla 1 – Tássio e João**
    * Análise manual do fluxo de trabalho (Branching Model)
    * Análise da estratégia de releases
* **Dupla 2 – Davi e Paulo**
    * Análise automatizada com `facebook/bart-large-mnli`
    * Classificação zero-shot
* **Dupla 3 – Luan e Matheus**
    * Análise com `Qwen/Qwen2.5-Coder-7B-Instruct`
    * Foco em governança e DevOps
* **Dupla 4 – Gabriel e Arthur**
    * Análise com `google/flan-t5-large`
    * Inferência guiada por instruções

*Todas as análises foram discutidas coletivamente em reuniões gerais para alinhamento e consolidação das conclusões.*

---

# 3. Tutorial de Utilização

## 3.1 Classificação Zero-Shot com facebook/bart-large-mnli
**Requisitos:**
* Python
* Biblioteca `transformers`

**Características:**
* Execução em CPU
* Não requer GPU

O modelo avalia a aderência semântica entre documentos do projeto e hipóteses pré-definidas (ex: Estratégia de Releases, Modelo de Branching). A saída consiste em uma lista de probabilidades indicando o grau de evidência de cada prática.

## 3.2 Análise Técnica com Qwen/Qwen2.5-Coder-7B-Instruct
**Requisitos:**
* GPU (Google Colab recomendado)
* Quantização em 4-bits

**Estrutura do Prompt:**
* **System Role:** Engenheiro de Software Sênior / DevOps
* **User Role:** Análise explícita da documentação

O modelo gera classificações acompanhadas de justificativas técnicas, que devem ser validadas manualmente.

## 3.3 Inferência Guiada com google/flan-t5-large
**Características:**
* Modelo leve
* Executável em CPU ou GPU
* Prompts curtos e restritivos

As respostas são diretas (ex: *GitHub Flow*, *Trunk-based Development*). Ideal para desempate entre análises e confirmação cruzada.

## 3.4 Execução Automatizada

Para facilitar a reprodução das análises, disponibilizamos um script unificado (`main.py`) que orquestra a execução dos modelos descritos acima.
O script (`main.py`) atua como o orquestrador do projeto. Ele permite selecionar o modelo de IA desejado e executar o pipeline de análise exatamente como foi realizado em nosso estudo, replicando os testes sobre o repositório.

### 3.4.1 Configuração do Ambiente

Recomendamos o uso de um ambiente virtual (`venv`) para isolar as dependências do projeto.

Passo 1: Criar e Ativar o Ambiente Virtual

**No Windows:**

```bash
# Cria o ambiente virtual
python -m venv venv

# Ativa o ambiente
.\venv\Scripts\activate
```
No Linux / macOS:

```Bash

# Cria o ambiente virtual
python3 -m venv venv

# Ativa o ambiente
source venv/bin/activate
````

Passo 2: Instalar Dependências
Com o ambiente ativo, instale as bibliotecas listadas no arquivo requirements.txt:

```Bash

pip install -r requirements.txt
```

### 3.4.2📂 Pré-requisitos e Estrutura de Pastas
Para que o script funcione corretamente (especialmente a Opção 1), é obrigatório que o repositório alvo da análise esteja clonado na raiz do projeto com o nome exato Jarvis.

A estrutura de diretórios deve seguir este padrão:

```Plaintext

.
├── main.py                 # Script principal de execução
├── requirements.txt        # Lista de dependências
├── analysis_bart/          # Módulo do modelo BART
├── analysis_qwen/          # Módulo do modelo Qwen
├── outputs/                # Pasta onde os resultados serão salvos
└── Jarvis/                 # O repositório Microsoft JARVIS clonado aqui
```

### 3.4.3 🚀 Como Executar
Certifique-se de que o ambiente virtual está ativo (vide seção 3.3.1).

Abra o terminal na raiz do projeto.

Execute o comando:

```Bash

python main.py
````
---

# 4. Identificação Manual da Governança

**Responsáveis:** João Henrique Britto Bomfim e Tassio Mateus de Carvalho

### Fluxo de Trabalho (Branching Model)
A análise do histórico de commits, utilizando a extensão Git Graph, revelou:
* Ausência da branch `develop`
* Centralização do desenvolvimento na branch `main`
* Uso de feature branches temporárias
* Integração via Pull Requests
* Ocorrência pontual de commits diretos na `main`

**Conclusão:** O projeto adota predominantemente o **GitHub Flow**, com pequenas aproximações ao Trunk-based Development.

### Estratégia de Releases
* Ausência de releases formais no GitHub
* Presença de changelog manual no README
* Atualizações vinculadas a:
    * Publicação de artigos científicos
    * Lançamento de benchmarks (TaskBench)
    * Ferramentas de apoio (EasyTool)

**Conclusão:** A estratégia de releases é **Ad-hoc / Research-based**, típica de projetos acadêmicos e experimentais.

---

# 5. Análise com Modelos de Linguagem

### facebook/bart-large-mnli
* **Resultado:** GitHub Flow | Tendência a Rapid Releases
* **Destaque:** Alta robustez e baixo risco de alucinação.

### Qwen/Qwen2.5-Coder-7B-Instruct
* **Resultado:** Rapid Releases | Gitflow
* **Limitação:** Justificativas inconsistentes em documentos com baixa densidade informacional.

### google/flan-t5-large
* **Resultado:** Releases Ad-hoc / Research-based | Trunk-based Development
* **Destaque:** Boa coerência conceitual com contexto acadêmico.

---

# 6. Comparação dos Resultados

| Modelo | Padrões Identificados | Efetividade |
| :--- | :--- | :---: |
| **Análise Manual** | **GitHub Flow + Research-based Releases** | **5** |
| BART MNLI | GitHub Flow + Rapid Releases | 4 |
| Qwen 2.5 Coder | Gitflow + Rapid Releases | 4 |
| FLAN-T5 Large | Trunk-based + Research-based | 4 |

---

# 7. Conclusão Geral

A análise evidenciou que o Microsoft JARVIS apresenta uma governança orientada à pesquisa, onde:

* O código evolui como extensão direta da produção científica.
* O versionamento serve à disseminação do conhecimento.
* A simplicidade operacional do GitHub Flow favorece experimentação rápida.
* As releases seguem marcos acadêmicos, não ciclos comerciais.

Os Modelos de Linguagem mostraram-se ferramentas eficazes para auditoria de governança, desde que utilizados com validação humana crítica, especialmente em projetos com documentação implícita.

> **Conclusão Final:** O JARVIS opera como um repositório vivo do estado da arte, alinhando práticas ágeis de desenvolvimento à dinâmica da pesquisa científica.
