# Pacote TCC — Angel Guillermo Morales Romero

Última atualização: 16/06/2026.
Todos os documentos foram nomeados em português, mantendo a funcionalidade do código.

## Conteúdo

### 1. Angel_TCC_Rev7.docx
Dissertação atualizada (88 páginas). Mudanças aplicadas nesta data:
- **Bibliografia**: 35 referências (4 correções definitivas):
  - SOUTO ANIDO corrigida (autoria dupla com García Rondón, GECONTEC vol. 4 n. 2 pp. 14-34, 2016)
  - HARWELL & BHATTARAI: título e URL corrigidos (Inside Amazon Go — the camera-filled...)
  - JIRAPOKAKUL removida (não verificável em fonte primária)
  - HERRERA LÓPEZ substituída por LIANG & WANG (1994) — DOI 10.1016/0377-2217(94)90119-8
- **Tabela 6.1, §5.3.3, §6.6 Q7**: valores REAIS medidos via PM4Py, com discurso
  reformulado como "validação de fidelidade estrutural":
  - Fitness 1,000 (todos os processos)
  - Precision 1,000 / 1,000 / 0,807
  - Generalization 0,978 / 0,978 / 0,973
  - Simplicity 1,000 / 1,000 / 0,867

Antes da entrega definitiva, abrir no Word e atualizar Sumário e Listas
(Ctrl+A + F9 ou aceitar o prompt automático).

### 2. CONTEXTO_DO_PROJETO.md
Documento de continuidade. Se este chat com o Claude ficar pesado ou for
cortado, abra um novo chat e cole este arquivo inteiro como primeira mensagem.

### 3. ESTRATEGIA_9_DIAS.md
Plano de trabalho dia a dia, de 16/06 até a entrega em 25/06/2026.

### 4. repositorio_github_tcc_senac_agmr/
Pasta com tudo o que precisa subir para o repositório GitHub. Pronta para
`git init && git push`. Passos detalhados em INSTRUCOES_GITHUB.md dentro dela.

**Estrutura (toda em português, exceto o que precisa permanecer em inglês
por convenção das ferramentas):**

    repositorio_github_tcc_senac_agmr/
    ├── LEIAME.md                   leia primeiro (era README.md)
    ├── INSTRUCOES_GITHUB.md        passo-a-passo para subir o repo
    ├── LICENCA                     licença MIT (com cabeçalho em PT)
    ├── CITACAO.cff                 ficha de citação acadêmica
    ├── requirements.txt            mantido (convenção pip)
    ├── .gitignore                  mantido (convenção git)
    ├── configuracoes/              hiperparâmetros (era configs/)
    │   └── default.yaml
    ├── exemplos/                   BPMN dos três processos (era examples/)
    │   ├── process_A.bpmn
    │   ├── process_B.bpmn
    │   └── process_C.bpmn
    ├── documentacao/               documentação técnica (era docs/)
    │   ├── ROADMAP.md
    │   └── CAPITULO_5_TEMPLATE.md
    ├── resultados/                 medições reais via PM4Py (era results/)
    │   ├── LEIAME.md
    │   ├── process_A.csv           event log gerado (728 KB)
    │   ├── process_B.csv           event log gerado (752 KB)
    │   ├── process_C.csv           event log gerado (1,1 MB)
    │   ├── pm4py_metrics.json      métricas exatas
    │   └── simulate_and_validate.py  script reprodutível
    └── src/                        pacote Python (MANTIDO em inglês)
        ├── bpmn_loader.py          imports do tipo "from src.X import Y"
        ├── case_generator.py
        ├── reward.py
        ├── workflow_env.py
        ├── gat_policy.py
        ├── baselines.py
        └── train.py

**Nota sobre os nomes em inglês mantidos:**
- `src/` e os arquivos `*.py` — renomear quebraria todos os imports Python
- `requirements.txt` — convenção pip; usuários esperam esse nome para
  rodar `pip install -r requirements.txt`
- `.gitignore` — exigência do Git
- Conteúdo dos arquivos Python — código em inglês por especificação da
  linguagem (def, class, return); docstrings e comentários estão em português

## Ordem sugerida ao abrir o pacote

1. Ler CONTEXTO_DO_PROJETO.md (10 minutos)
2. Ler ESTRATEGIA_9_DIAS.md (5 minutos)
3. Abrir Angel_TCC_Rev7.docx
4. No fim de semana, entrar em repositorio_github_tcc_senac_agmr/INSTRUCOES_GITHUB.md

## O que ainda está pendente

- 5 diagramas conceituais (Figuras 1-5) — próxima frente a executar
- Tabelas 6.2 a 6.5 (resultados PPO+GAT) — decisão entre treino real reduzido
  ou reformulação como projeção honesta
- Frases >40 palavras (#26 do coordenador) — polimento final
- Criação do repositório GitHub no fim de semana
- Atualização do Sumário e Listas no Word na hora de gerar PDF final
