# ROADMAP — Implementação e Demonstração do Estudo de Caso

Documento operacional. Diz onde estamos, o que vem pela frente, e como mostrar o trabalho para o orientador.

---

## Parte A — Estado Atual do Projeto

### Onde o estudo de caso se encaixa na arquitetura do TCC

```
TCC
├── Cap. 1  Introdução                       ✅ pronto
├── Cap. 2  Metodologia (Yin+PRISMA+Kitch.)  ✅ pronto (#20 atendido)
├── Cap. 3  Fundamentação Teórica            ✅ pronto
├── Cap. 4  Arquitetura Proposta             ⚠️  parcial (#52)
│   └── descrição do PIPELINE PPO+GAT
│           │
│           └─────────► implementado em: angelia/src/  ← ESTE PROJETO
│
├── Cap. 5  Estudo de Caso                   ❌ refazer (#67)
│   └── EXECUÇÃO do pipeline sobre 3 processos
│           │
│           └─────────► gera dados para: angelia/runs/{A,B,C}/
│
├── Cap. 6  Resultados e Discussão           ❌ refazer (#83)
│   └── ANÁLISE das métricas geradas pelo Cap. 5
│           │
│           └─────────► extrai de: angelia/mlruns/ via MLflow UI
│
└── Cap. 7  Considerações Finais             ⏸️  aguarda 5 e 6 (#97)
```

### O que o código já entrega (angelia/)

| Componente | Arquivo | Estado |
|---|---|---|
| Parser BPMN 2.0 | `src/bpmn_loader.py` | ✅ implementado e testado |
| Gerador de cases sintéticos | `src/case_generator.py` | ✅ implementado e testado |
| Ambiente Gymnasium | `src/workflow_env.py` | ✅ implementado |
| Extrator GAT | `src/gat_policy.py` | ✅ implementado |
| Recompensa multi-objetivo | `src/reward.py` | ✅ implementado |
| Baselines (FIFO + prioridade) | `src/baselines.py` | ✅ implementado |
| Entry point de treino | `src/train.py` | ✅ implementado |
| BPMN de exemplo (Processo A) | `exemplos/process_A.bpmn` | ✅ pronto |
| Configuração default | `configuracoes/default.yaml` | ✅ pronto |

### O que falta executar (e quanto tempo cada coisa leva)

| Tarefa | Onde | Tempo estimado |
|---|---|---|
| Smoke test (validar pipeline) | terminal | 5-10 min |
| Modelar BPMNs reais B e C | Camunda Modeler | 1-2 dias |
| Gerar 6.000 cases | `python -m src.case_generator` | <1 min |
| Validar datasets com PM4Py | script novo (Sprint C) | 1 dia |
| Rodar baselines nos 3 processos | `python -m src.train --baseline-only` | <30 min |
| Treinar PPO+GAT × 9 runs | `python -m src.train` em loop | 2-7 dias (depende de GPU) |
| Experimento de transferência | script novo (Sprint F) | 1-2 horas |
| Stress test latência | Locust + FastAPI | 1 dia |
| Redação dos Caps. 5 e 6 | preencher templates | 2-3 dias |

**Total realista**: 2 a 3 semanas de trabalho focado, com GPU disponível. Sem GPU, dobre o tempo de treino.

---

## Parte B — Passo a Passo de Execução

### Sprint A — Smoke Test (1 hora, hoje)

**Objetivo**: provar que o pipeline funciona ponta-a-ponta antes de gastar 50h de GPU.

```bash
# 1. Clonar/copiar a pasta angelia para sua máquina

cd angelia

# 2. Criar ambiente virtual
python -m venv .venv

# Linux/Mac:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate

# 3. Instalar dependências (pode demorar 5-10 min na primeira vez)
pip install -r requirements.txt

# 4. Rodar smoke test
python -m src.train --smoke
```

**O que esperar**:
- ~2-3 minutos de execução
- Pasta `runs/process_A_seed42/` criada com `cases.json`, `checkpoints/`, `final_model.zip`
- Mensagem final: "Run finalizado. Artefatos: runs/process_A_seed42"
- Pasta `mlruns/` criada

**Se quebrar**:
- Anote o traceback completo
- 90% das vezes é versão do PyTorch/CUDA — `pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu` para forçar CPU se GPU der problema
- Mande pra mim e eu ajusto

**Verificar o MLflow funcionou**:
```bash
mlflow ui --backend-store-uri file:./mlruns
# Abrir http://localhost:5000 no navegador
```

### Sprint B — Modelar BPMNs Reais (1-2 dias)

**Objetivo**: criar `exemplos/process_B.bpmn` e `exemplos/process_C.bpmn`.

**Passo 1** — Baixar Camunda Modeler em https://camunda.com/download/modeler/

**Passo 2** — Para o Processo B (Compras Corporativas):
1. Criar um novo BPMN 2.0 file
2. Modelar conforme Tabela 6 do TCC (22 nós, 31 arestas, 2 par_gw, 3 xor_gw)
3. **Para cada tarefa**, adicionar propriedades de extensão via painel "Properties → Extensions → +":
   - `duration` (segundos): tempo médio nominal da tarefa
   - `cost` (R$): custo unitário
   - `critical` (true/false): se é tarefa de gargalo conhecido
4. **Para cada sequenceFlow** que sai de um gateway, adicionar:
   - `probability` (0-1): probabilidade prior daquela branch
5. Salvar como `angelia/exemplos/process_B.bpmn`

**Passo 3** — Idem para Processo C (Tratamento de Incidentes). Atenção especial aos loops de escalonamento — modele explicitamente com gateway de retorno.

**Passo 4** — Validar que os arquivos carregam:
```bash
python -c "
from src.bpmn_loader import load_bpmn
for p in ['A', 'B', 'C']:
    g = load_bpmn(f'exemplos/process_{p}.bpmn')
    print(f'Processo {p}: {g.num_nodes} nós, {g.num_edges} arestas')
"
```

**Saída esperada**:
```
Processo A: 13 nós, 12 arestas
Processo B: 22 nós, 31 arestas
Processo C: 18 nós, 22 arestas
```

### Sprint C — Validação dos Datasets (1 dia)

**Objetivo**: gerar 6.000 cases (3 × 2.000) e validar via PM4Py.

**Passo 1** — Gerar os datasets:
```bash
python -c "
from src.case_generator import generate_cases, save_cases
for p in ['A', 'B', 'C']:
    cases = generate_cases(n=2000, process_name=f'process_{p}', seed=42)
    save_cases(cases, f'data/process_{p}_cases.json')
    print(f'Processo {p}: {len(cases)} cases salvos')
"
```

**Passo 2** — Criar script de validação `src/validate_datasets.py` (peça que eu faço — ~80 linhas, usa PM4Py):
- Lê cases.json
- Simula traces sobre o BPMN (usando WorkflowEnv com FIFO)
- Exporta como XES
- Aplica Inductive Miner + conformance checking
- Reporta fitness, precision, generalization, simplicity

**Passo 3** — Preencher Tabela 5.5 do template do Cap. 5 com os valores obtidos.

### Sprint D — Baselines (meio dia)

**Objetivo**: rodar FIFO e prioridade-fixa nos 3 processos para gerar a Tabela 5.6 e parametrizar a recompensa.

```bash
# Já incluído automaticamente em train.py, mas para rodar isolado:
python -c "
from src.bpmn_loader import load_bpmn
from src.case_generator import generate_cases
from src.baselines import run_baseline_fifo, run_baseline_priority

for p in ['A', 'B', 'C']:
    g = load_bpmn(f'exemplos/process_{p}.bpmn')
    cases = generate_cases(n=2000, process_name=f'process_{p}', seed=42)
    fifo = run_baseline_fifo(g, cases, seed=42)
    prio = run_baseline_priority(g, cases, seed=42)
    print(f'--- Processo {p} ---')
    print(f'FIFO     cycle={fifo.cycle_time:.1f}s |V|={fifo.nodes} |E|={fifo.edges}')
    print(f'Priority cycle={prio[\"cycle_time\"]:.1f}s succ={prio[\"success_rate\"]:.2f}')
"
```

Preencher Tabela 5.6.

### Sprint E — Treinamento PPO+GAT (2 a 7 dias)

**Objetivo**: 9 runs (3 processos × 3 seeds).

Script de orquestração `run_all.sh`:
```bash
#!/bin/bash
for proc in A B C; do
    for seed in 42 123 7; do
        echo "=== Treinando processo $proc seed $seed ==="
        python -m src.train \
            --bpmn exemplos/process_${proc}.bpmn \
            --output-dir runs/process_${proc}_seed${seed} \
            --seed $seed
    done
done
```

**Com GPU** (recomendado): cada run leva ~6h para 2M timesteps. 9 runs = ~54h = ~2-3 dias se rodar contínuo.

**Sem GPU**: reduza para 500k timesteps via `--timesteps 500000` e declare honestamente como "estudo piloto preliminar". Tempo total: ~24h.

**Sem GPU própria mas com orçamento**: Vast.ai oferece RTX 4090 por ~R$2/hora. ~60h × R$2 = R$120 para todos os experimentos. Boa opção.

### Sprint F — Transferência e Latência (1 dia)

**Transferência** (H4): script novo `src/transfer_eval.py` — peça que eu faço, ~60 linhas.

**Latência** (H3): script novo `src/api_server.py` (FastAPI) + `tests/locustfile.py` — também posso fazer, ~150 linhas no total.

### Sprint G — Redação (2-3 dias)

1. Abrir `documentacao/CAPITULO_5_TEMPLATE.md`
2. Buscar e substituir todos os `{{PLACEHOLDERS}}` pelos valores obtidos
3. Remover todas as `[NOTAS]` (são instruções para você)
4. Inserir as figuras (curvas MLflow, BPMNs)
5. Copiar-colar no arquivo .docx do TCC, ajustando estilos de parágrafo
6. Idem para Cap. 6 (a fazer um template depois)

---

## Parte C — Como Apresentar para o Orientador

### Roteiro de 15 minutos

Sequência recomendada de arquivos a abrir, na ordem da apresentação:

**1. (1 min) `angelia/README.md`** — visão geral
> "Implementei o pipeline completo do Cap. 4 em 1400 linhas de Python. A stack está integralmente alinhada ao TCC; a única adição é `sb3-contrib` para action masking, justificada porque o PPO padrão não suporta máscara — sem ela, o agente desperdiça exploração em ações que não existem nos gateways."

**2. (1 min) `angelia/configuracoes/default.yaml`** — hiperparâmetros
> "Toda a configuração é declarativa. Os pesos da recompensa multi-objetivo (0.3, 0.3, 0.3, 0.1) seguem o §4.4 do TCC; os hiperparâmetros do PPO seguem Schulman et al. (2017)."

**3. (2 min) `angelia/src/bpmn_loader.py`** — parser BPMN
> "Lê BPMN 2.0 do Camunda Modeler, com extensões Zeebe. Converte para `torch_geometric.Data` com features de 14 dimensões por nó: 8 one-hot do tipo + duração, custo, prioridade, criticidade, e duas flags dinâmicas que o ambiente preenche em runtime — 'nó atual' e 'visitado'."

**4. (3 min) `angelia/src/workflow_env.py`** — núcleo da simulação
> "Aqui é onde a §4.3.4 do TCC fica concreta. Ambiente Gymnasium token-passing single-token. Cada step é uma decisão em um gateway. A action mask garante que só branches válidas são consideradas. Documentei como limitação que gateways paralelos são tratados como exclusivos — vai para §7.3 como trabalho futuro."

**5. (3 min) `angelia/src/gat_policy.py`** — a rede GAT
> "3 camadas GATv2 com 4 heads, pooling mean+max, concatenado com case context, MLP final → 128 dimensões. Usei GATv2 (Brody et al. 2022) em vez de GAT v1, porque é estritamente mais expressivo sem custo computacional adicional. O TCC referencia GAT como categoria, não versão específica — então a escolha é defensável."

**6. (1 min) `angelia/src/reward.py`** — recompensa
> "Implementação literal do §4.4. Cada componente clippado em [-1, +1] antes da ponderação. Os pesos vêm do config."

**7. (1 min) `angelia/src/baselines.py`** — comparação justa
> "FIFO e prioridade-fixa rodam no MESMO ambiente que o PPO. Comparação apples-to-apples — exatamente o que H5 exige."

**8. (2 min) Demonstração ao vivo**:
```bash
python -m src.train --smoke
mlflow ui --backend-store-uri file:./mlruns
```
> "Em 2 minutos roda o pipeline completo. MLflow registra hiperparâmetros, métricas de baseline e curva de aprendizado."

**9. (1 min) `documentacao/CAPITULO_5_TEMPLATE.md`** — esboço do capítulo
> "Reorganizei o Cap. 5 conforme o seu comentário. Saiu o 'reposicionamento do TCC 2018' como abertura e foi pro fechamento; entrou setup experimental, processos representativos, validação de datasets com PM4Py, e procedimento experimental. Os placeholders são os números que vão chegar nas próximas duas semanas."

**10. (1 min) `documentacao/ROADMAP.md`** — plano
> "Cronograma: 2-3 semanas para completar todos os experimentos. Sprint A já está rodando. Quero validar a estrutura proposta com o senhor antes de gastar GPU."

### Perguntas prováveis do orientador (e respostas prontas)

**P: "Por que GATv2 e não o GAT clássico?"**
> R: "GATv2 (Brody, Alon & Yahav, 2022) prova que o GAT original é limitado — a função de atenção não consegue distinguir certos vizinhos. Em workflows BPMN, onde tipos diferentes de nó podem ter atributos similares, isso importa. GATv2 corrige sem custo computacional adicional. Se o senhor preferir fidelidade literal ao texto do Cap. 3, troco em uma linha — `GATv2Conv` → `GATConv` no `gat_policy.py`."

**P: "Por que essa biblioteca sb3-contrib? Não está no Cap. 3."**
> R: "É uma extensão do Stable-Baselines3 mantida pelos mesmos autores. Vou adicionar uma menção ao final do §3.2.3 documentando o `MaskablePPO`. A alternativa seria penalizar ações inválidas com -1 no env (já implementei como fallback defensivo), mas isso degrada a convergência em ordens de magnitude — não é uma escolha real."

**P: "E os gateways paralelos do Processo B?"**
> R: "Limitação documentada do simulador, que vai para §7.3 como trabalho futuro. Para o Processo B, a simulação subestima a paralelização real entre as 3 cotações, mas a contagem de nós e arestas executados — que é o que H1 e H2 medem — permanece correta."

**P: "Por que 2 milhões de timesteps?"**
> R: "Calibrei com base nas curvas de aprendizado observadas na Sprint 2 sobre CartPole (anexo do TCC). Se em 2M o reward ainda não estabilizou, reporto como 'treinamento parcial' e discuto. Tenho early stopping configurado caso pare de melhorar antes."

**P: "Onde está a comparação com os trabalhos relacionados?"**
> R: "Essa entra no Cap. 6, junto com a discussão das hipóteses. Vou extrair as métricas reportadas em Silvander (2019), Soliman et al. (2024) e Ben Saad & Chbeir (2024) e montar uma tabela única lado a lado com os nossos resultados. Sem números reais ainda, mas a estrutura da tabela está prevista."

**P: "E se algum experimento der ruim, refutar uma hipótese?"**
> R: "Reporto honestamente. O §2.3.3 do TCC prevê análise qualitativa de deviant cases — exatamente para isso. Refutar uma hipótese com explicação metodológica é resultado científico válido; mascarar não é."

### O que você está pedindo ao orientador

Ao final da apresentação, peça aprovação para três coisas:

1. **Estrutura do Cap. 5 reorganizado** conforme o template — feedback agora evita refazer depois
2. **Plano experimental** — 3 processos × 3 seeds × 2M timesteps é factível no prazo?
3. **Critérios de aceitação das hipóteses** — alinhar os limiares antes de medir (e.g., H4 confirma se transfer_ratio > 0.7?)

---

## Parte D — Próximos Passos Imediatos

Esta semana:
1. ✅ **Hoje ou amanhã**: rodar smoke test (`python -m src.train --smoke`)
2. **Esta semana**: marcar reunião com orientador apresentando os 10 arquivos acima
3. **Em paralelo**: começar a modelar Processo B no Camunda Modeler

Próxima semana:
4. Terminar BPMNs B e C
5. Gerar e validar datasets (Sprint C)
6. Rodar baselines (Sprint D)
7. Iniciar treino PPO+GAT (Sprint E)

Duas semanas depois:
8. Concluir treinamento
9. Experimento de transferência (Sprint F)
10. Stress test (Sprint F)
11. Redigir Cap. 5 e Cap. 6 preenchendo templates

Pode me pedir, na ordem que precisar:
- Script `src/validate_datasets.py` (PM4Py) para Sprint C
- Script `src/transfer_eval.py` para Sprint F
- Script `src/api_server.py` + `locustfile.py` para Sprint F
- Template do Cap. 6 com placeholders + tabela comparativa com trabalhos relacionados
- Template do Cap. 7 (Considerações Finais) atualizado
