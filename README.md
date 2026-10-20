# tcc_senac_agmr — pipeline PPO+GAT para otimização de workflows BPMN

Código de apoio ao TCC *"Otimização de Workflows Empresariais com Machine Learning: uma arquitetura PPO+GAT integrada ao Camunda Platform 8"*, de **Angel Guillermo Morales Romero**, apresentado na **pós-graduação lato sensu em Inteligência Artificial do Centro Universitário Senac** (São Paulo, 2026). Implementa o estudo de caso descrito no Capítulo 5.

*In English:* code for my postgraduate thesis (Senac, Brazil, 2026). A reinforcement-learning agent (Maskable PPO) with a Graph Attention Network policy learns to route cases through BPMN 2.0 workflows exported from Camunda 8, compared against FIFO and fixed-priority baselines. Structural fidelity of the simulated processes was validated with PM4Py.

## O que este código entrega

- **Parser BPMN 2.0** (Camunda 8 / Zeebe) → grafo direcionado PyTorch Geometric
- **Gerador de cases sintéticos** com distribuição 70/20/10 (normais / gargalos / exceções)
- **Ambiente Gymnasium** que simula execução de workflows token-passing
- **Rede GAT de 3 camadas e 4 heads** integrada ao Stable-Baselines3
- **Agente PPO mascarado** (sb3-contrib MaskablePPO) com pesos multi-objetivo
- **Baselines FIFO e prioridade-fixa** para comparação direta (H5 do TCC)
- **Logging MLflow + W&B** para tracking de experimentos

## Quickstart

```bash
# 1. Ambiente
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Smoke test (~2 min em CPU) — valida pipeline ponta a ponta
python -m src.train --smoke

# 3. Treino completo em Processo A (~12-18h em GPU; ~4-6h se reduzir para 500k timesteps)
python -m src.train --config configuracoes/default.yaml

# 4. Treino nos outros dois processos
python -m src.train --bpmn exemplos/process_B.bpmn --output-dir runs/process_B
python -m src.train --bpmn exemplos/process_C.bpmn --output-dir runs/process_C

# 5. Inspecionar runs
mlflow ui --backend-store-uri file:./mlruns
```

## Estrutura

```
tcc_senac_agmr/
├── README.md                  este arquivo
├── CITACAO.cff                como citar (TCC e software)
├── LICENCA                    licença MIT
├── requirements.txt           dependências pinned para reprodutibilidade
├── configuracoes/
│   └── default.yaml           hiperparâmetros (PPO, GAT, recompensa)
├── exemplos/
│   ├── process_A.bpmn         Processo A (Seleção de Pessoal)
│   ├── process_B.bpmn         Processo B
│   └── process_C.bpmn         Processo C
├── resultados/                validação via PM4Py (métricas e script)
└── src/
    ├── bpmn_loader.py         BPMN XML → torch_geometric.Data
    ├── case_generator.py      cases sintéticos (Faker + numpy)
    ├── reward.py              R = w_T·R_tempo + w_C·R_complex + w_Q·R_qual + w_S·R_sucesso
    ├── workflow_env.py        ambiente Gymnasium (simulação token-passing)
    ├── gat_policy.py          extrator GAT para SB3 MaskablePPO
    ├── baselines.py           FIFO e prioridade-fixa
    └── train.py               entry point com MLflow
```

## Stack tecnológico — alinhamento com o TCC

Todas as bibliotecas usadas aqui estão **explicitamente listadas** no TCC. Não
há migração silenciosa. A única adição é `sb3-contrib`, justificada abaixo.

| Função no pipeline | Biblioteca | Versão | Onde aparece no TCC |
|---|---|---|---|
| Deep learning | PyTorch 2.4 | `torch==2.4.0` | §4.2.2, §4.3.2 |
| GNN | PyTorch Geometric | `torch-geometric==2.5.3` | §3.3, §4.3.2 |
| RL framework | Stable-Baselines3 | `stable-baselines3==2.3.2` | §3.2.3, §4.2.3.2 |
| Action masking | sb3-contrib | `sb3-contrib==2.3.0` | **(adição — ver abaixo)** |
| Env interface | Gymnasium | `gymnasium==0.29.1` | §4.3.4 |
| Parsing BPMN | lxml | `lxml==5.2.2` | §4.3.1 |
| Grafos | NetworkX | `networkx==3.3` | §4.3.1 |
| Process Mining | PM4Py | `pm4py==2.7.11` | §5.3, §6.1 |
| Geração sintética | Faker | `faker==26.0.0` | §5.3 |
| Experiment tracking | MLflow | `mlflow==2.16.0` | §4.2.3.2, §5.2 |
| Visualização | W&B (opcional) | `wandb==0.17.6` | §4.2.3.2 |
| Config | PyYAML | `pyyaml==6.0.2` | (convenção) |

### Por que `sb3-contrib` (a única adição)?

Stable-Baselines3 **não** implementa action masking nativamente em PPO. Em
BPMN, em cada gateway o agente só tem disponíveis as branches que efetivamente
saem daquele nó — tipicamente 2 a 4 de até MAX_BRANCHES=8 possíveis. Sem
máscara, o PPO desperdiça grande parte da exploração amostrando ações
inexistentes, o que deteriora a velocidade de convergência em ordens de
magnitude e produz políticas inconsistentes.

`sb3-contrib` é o pacote oficial de extensões mantido **pelos mesmos autores**
do `stable-baselines3` (Antonin Raffin, Adam Gleave et al.). `MaskablePPO`
expõe a mesma API do `PPO` original — a diferença é a aceitação de
`action_masks()` no env. Não é uma migração de framework; é um drop-in que
preserva todas as garantias do SB3 (reprodutibilidade, callbacks,
checkpointing, MLflow/TensorBoard).

> Sem `sb3-contrib`, a alternativa é tratar ações inválidas com penalidade
> alta no env (já implementado defensivamente: -1.0 + terminação). Funciona,
> mas converge significativamente mais devagar.

### Decisão menor: `GATv2Conv` em vez de `GATConv`

`src/gat_policy.py` usa `GATv2Conv` (Brody, Alon & Yahav, 2022 — *"How
attentive are graph attention networks?"*). É estritamente mais expressivo
que o GAT v1 original (Veličković et al., 2018) e tornou-se a recomendação
default do PyTorch Geometric para novos projetos. Como o TCC referencia
"Graph Attention Networks (GAT)" como **categoria** (não como versão
específica), GATv2 é uma escolha academicamente correta e mais robusta.

Trocar de volta para GAT v1 é trivial: substituir `GATv2Conv` por `GATConv`
em `src/gat_policy.py` — a interface é idêntica.

## Como integrar com seus BPMN reais (Camunda Modeler)

1. No Camunda Modeler, exporte o processo como `.bpmn` (BPMN 2.0 XML).
2. Em cada **task** (userTask, serviceTask), adicione propriedades de extensão:
   ```xml
   <bpmn:extensionElements>
     <zeebe:properties>
       <zeebe:property name="duration" value="180"/>  <!-- segundos -->
       <zeebe:property name="cost" value="50"/>
       <zeebe:property name="critical" value="true"/>
     </zeebe:properties>
   </bpmn:extensionElements>
   ```
3. Em cada **sequenceFlow** que sai de um gateway, opcionalmente adicione
   `probability` (prior usado pelo baseline de prioridade-fixa).
4. Rode `python -m src.train --bpmn meu_processo.bpmn --output-dir runs/meu`.

Se uma propriedade não estiver presente, o loader aplica defaults
configuráveis (`default_duration_s=60`, `default_cost=1.0`, `priority=0.5`).

## Como ler os resultados

O comando `mlflow ui --backend-store-uri file:./mlruns` abre uma interface
local na porta 5000. Para cada run você verá:

- **Hiperparâmetros**: configuração completa do treino
- **Métricas de baseline**: cycle time, |V|, |E| do FIFO e prioridade-fixa
  (esses são os números contra os quais H1, H2, H5 serão avaliadas)
- **Métricas de treino** (a cada `n_steps`): rollout reward, value loss,
  policy loss, KL-divergence, entropy, FPS
- **Métricas de avaliação** (a cada `eval_freq` timesteps): mean reward
  determinístico sobre `n_eval_episodes`
- **Artefatos**: `final_model.zip`, `cases.json`, `summary.txt`

## Limitações conhecidas

1. **Gateways paralelos modelados como exclusivos**: a simulação é
   single-token. Para Processo B (compras corporativas) com par_gw genuínos,
   essa simplificação subestima a paralelização real. Trabalho futuro:
   simulador token-passing concorrente (citar §7.3 do TCC).

2. **Loops no Processo C tratados com guard máximo** de `max_nodes * 4`
   transições por episódio. Suficiente para os SLAs típicos de incidentes,
   mas mereceria modelagem explícita de SLA temporizado.

3. **Qualidade da branch é heurística sintética** baseada em prioridade do
   case e probabilidade prior da branch. Em datasets reais (XES via PM4Py),
   substituir por métrica derivada do log (e.g., conformance fitness).

## Validação estrutural (PM4Py)

A pasta `resultados/` traz as métricas medidas com PM4Py sobre os event logs simulados dos três processos (fitness, precision, generalization e simplicity) e o script para reproduzi-las. Ver `resultados/LEIAME.md`.

## Autoria e citação

Autor: **Angel Guillermo Morales Romero** — [LinkedIn](https://www.linkedin.com/in/angel-guillermo-morales-romero) · [portfólio](https://aguilhermemr.github.io).

Para citar este trabalho, use os dados de `CITACAO.cff`. Código sob licença MIT (`LICENCA`).
