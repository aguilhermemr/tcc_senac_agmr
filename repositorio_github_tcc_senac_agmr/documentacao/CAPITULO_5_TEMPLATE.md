# CAPÍTULO 5 — TEMPLATE REORGANIZADO

> **Como usar este template**
>
> - Placeholders entre `{{CHAVES_DUPLAS}}` são valores a preencher após executar os experimentos da seção 5.4.
> - Trechos entre `[NOTA: ...]` são instruções para você — apagar antes de colar no docx.
> - Trechos entre `[FIGURA X — ...]` e `[TABELA X — ...]` marcam onde inserir figuras/tabelas (com legenda ABNT abaixo).
> - Pode copiar-colar para o Word; o estilo de parágrafo do seu TCC será mantido. Adapte numeração de tabelas/figuras à contagem global do documento.
> - Citações usam formato (Autor, Ano) conforme ABNT NBR 10520. As referências de PM4Py, Camunda 8, Stable-Baselines3, sb3-contrib, Yin, van der Aalst, Brody et al. já existem na sua bibliografia do Cap. 2/3 — apenas reaproveitar.

---

# 5 ESTUDO DE CASO

Este capítulo descreve o estudo de caso conduzido para avaliar empiricamente a arquitetura proposta no Capítulo 4. A apresentação segue a estrutura metodológica do estudo de caso embedded de Yin (2018), tratando cada um dos três processos representativos como uma unidade de análise dentro do caso único "otimização de workflows BPMN com PPO+GAT" (componente 6 da estrutura de Yin, conforme §2.2.6). A seção 5.1 detalha a configuração experimental; 5.2 caracteriza os processos de negócio; 5.3 descreve a geração e validação dos datasets sintéticos; 5.4 apresenta o procedimento experimental e seus resultados de baseline; e 5.5 contrasta a abordagem atual com o trabalho prévio do autor (Morales Romero, 2018).

## 5.1 Configuração Experimental

### 5.1.1 Ambiente de Hardware e Software

Os experimentos foram executados em ambiente {{TIPO_AMBIENTE: local | nuvem | misto}} com as seguintes especificações:

- CPU: {{CPU_MODELO}} ({{N_CORES}} núcleos físicos)
- GPU: {{GPU_MODELO}} ({{GPU_VRAM_GB}} GB VRAM) — utilizada para treinamento PPO
- Memória RAM: {{RAM_GB}} GB
- Sistema operacional: {{OS_VERSAO}}
- Python: 3.11

As versões das bibliotecas foram fixadas para garantir reprodutibilidade, conforme princípio metodológico estabelecido em §2.2.4 (van der Aalst, 2016). O Quadro 5.1 apresenta as principais dependências.

[TABELA 5.1 — Stack tecnológico utilizado]

| Função | Biblioteca | Versão | Referência no documento |
|---|---|---|---|
| Deep learning | PyTorch | 2.4.0 | §3.2.2 |
| Graph Neural Networks | PyTorch Geometric | 2.5.3 | §3.3 |
| Reinforcement Learning | Stable-Baselines3 | 2.3.2 | §3.2.3 |
| Action masking | sb3-contrib | 2.3.0 | §4.3.3 |
| Interface de ambientes | Gymnasium | 0.29.1 | §4.3.4 |
| Parsing BPMN | lxml | 5.2.2 | §4.3.1 |
| Análise de grafos | NetworkX | 3.3 | §4.3.1 |
| Process Mining | PM4Py | 2.7.11 | §5.3.3 |
| Geração de dados sintéticos | Faker | 26.0.0 | §5.3.1 |
| Tracking de experimentos | MLflow | 2.16.0 | §4.3.5 |

Fonte: O autor (2026).

### 5.1.2 Reprodutibilidade

Para assegurar a reprodutibilidade dos resultados empíricos, foram adotadas quatro práticas:

1. **Versionamento integral do código** em repositório Git (commit hash `{{COMMIT_HASH}}`), incluindo a totalidade dos scripts de geração de dados, treinamento, avaliação e exportação de métricas.
2. **Sementes aleatórias fixas**: foram utilizadas as sementes 42, 123 e 7 — três runs independentes por processo, totalizando nove treinamentos completos. A média e o desvio-padrão entre seeds compõem os intervalos de confiança reportados em §6.
3. **Datasets regeneráveis a partir do código**: nenhum arquivo de dados é versionado em binário; os 6.000 cases sintéticos podem ser regenerados em poucos segundos a partir do código-fonte e das sementes declaradas.
4. **Configuração declarativa**: todos os hiperparâmetros encontram-se em arquivo YAML único (`configuracoes/default.yaml`), eliminando "magic numbers" espalhados pelo código.

### 5.1.3 Hiperparâmetros do Agente

O Quadro 5.2 reúne os hiperparâmetros do PPO, calibrados conforme recomendações de Schulman et al. (2017) e validação prévia sobre CartPole-v1 (§4.3.3). O Quadro 5.3 apresenta a arquitetura do extrator GAT, derivada das diretrizes de Veličković et al. (2018) e Brody et al. (2022).

[TABELA 5.2 — Hiperparâmetros do PPO]

| Parâmetro | Valor | Justificativa |
|---|---|---|
| total_timesteps | {{TIMESTEPS_TOTAL}} | {{JUST_TIMESTEPS}} |
| n_envs (paralelos) | 4 | balanceamento entre paralelismo e overhead |
| n_steps por rollout | 2048 | recomendação padrão Schulman et al. (2017) |
| batch_size | 64 | 32 mini-batches por rollout |
| n_epochs | 10 | clipping evita overfitting do rollout |
| gamma (desconto) | 0.99 | episódios curtos (≤200 steps) |
| gae_lambda | 0.95 | redução de variância na vantagem |
| learning_rate | 3e-4 | default empiricamente robusto |
| clip_range | 0.2 | recomendação do paper original |
| ent_coef | 0.0 | exploração suficiente via stochasticity da política |
| vf_coef | 0.5 | equilíbrio actor/critic |
| max_grad_norm | 0.5 | estabilização de gradiente |

Fonte: O autor (2026), adaptado de Schulman et al. (2017).

[TABELA 5.3 — Arquitetura do extrator GAT]

| Camada | Parâmetros |
|---|---|
| Input | 14 features de nó (8 one-hot tipo + 6 numéricas) |
| GATv2Conv #1 | 4 heads, 64 unidades/head, dropout 0.2, ELU |
| GATv2Conv #2 | 4 heads, 64 unidades/head, dropout 0.2, ELU |
| GATv2Conv #3 | 1 head, 64 unidades, dropout 0.2 |
| Pooling | concat(global_mean_pool, global_max_pool) → 128 |
| Concat | + case_context (5 dims) → 133 |
| MLP de projeção | 128 → ReLU → 128 |

Fonte: O autor (2026).

> [NOTA: Justificativa para usar GATv2 em vez de GAT v1, caso o orientador questione:
> "Foi adotada a variante GATv2 (Brody, Alon & Yahav, 2022), que demonstrou-se estritamente mais expressiva que o GAT original (Veličković et al., 2018) sem custo computacional adicional significativo. A categoria 'Graph Attention Networks' permanece, conforme descrito em §3.3.2."]

## 5.2 Processos de Negócio Representativos

A seleção dos três processos seguiu três critérios convergentes: (i) heterogeneidade estrutural (linear, paralelo, cíclico); (ii) cobertura dos quatro padrões BPMN mais frequentes em workflows corporativos (Dijkman, La Rosa & Reijers, 2012); (iii) viabilidade de modelagem completa pelo autor no Camunda Modeler 5.x, sem dependência de logs proprietários.

### 5.2.1 Processo A — Seleção de Pessoal

[NOTA: Manter o texto existente do §5.2 do TCC. Ajustar contagens conforme o arquivo BPMN final.]

[FIGURA 5.1 — Diagrama BPMN do Processo A]

Características estruturais:
- Total de nós: {{NODES_A}} (incluindo eventos de início e fim)
- Total de arestas: {{EDGES_A}}
- Gateways exclusivos (xor): {{XOR_A}}
- Gateways paralelos (par): 0
- Gateways inclusivos (inc): 0
- Profundidade máxima do grafo: {{DEPTH_A}}
- Duração nominal média estimada: {{DUR_NOM_A}} segundos

### 5.2.2 Processo B — Compras Corporativas

[FIGURA 5.2 — Diagrama BPMN do Processo B]

Características estruturais:
- Total de nós: {{NODES_B}}
- Total de arestas: {{EDGES_B}}
- Gateways exclusivos (xor): {{XOR_B}}
- Gateways paralelos (par): {{PAR_B}}
- Gateways inclusivos (inc): {{INC_B}}
- Profundidade máxima do grafo: {{DEPTH_B}}
- Duração nominal média estimada: {{DUR_NOM_B}} segundos

> [NOTA: Documentar aqui a limitação do simulador: gateways paralelos são modelados como exclusivos na presente versão (justificativa em §7.3). Para o Processo B, isso significa que a simulação não captura a paralelização real entre as 3 cotações de fornecedores; o tempo é subestimado, mas a contagem de nós/arestas executados é correta.]

### 5.2.3 Processo C — Tratamento de Incidentes

[FIGURA 5.3 — Diagrama BPMN do Processo C]

Características estruturais:
- Total de nós: {{NODES_C}}
- Total de arestas: {{EDGES_C}}
- Gateways exclusivos (xor): {{XOR_C}}
- Gateways paralelos (par): 0
- Gateways inclusivos (inc): {{INC_C}}
- Loops (escalonamento): {{LOOPS_C}}
- Profundidade máxima do grafo: {{DEPTH_C}}

A presença de loops no Processo C exige o tratamento via guard máximo de transições por episódio (4 × |V| no simulador), evitando episódios infinitos quando o agente entra em ciclos de reabertura. Esse limite excede em ordem de magnitude o pior caso observado em logs de helpdesk reais (BPI Challenge 2014, máximo 8 reaberturas).

## 5.3 Geração e Validação dos Datasets Sintéticos

### 5.3.1 Estratégia de Geração

Foram gerados 6.000 cases sintéticos no total — 2.000 por processo — com distribuição categórica controlada, refletindo a frequência observada em event logs corporativos públicos (BPI Challenge 2012, 2014, 2017; Helpdesk; Sepsis):

- **70% cases normais**: percorrem o "happy path" predominante, com atributos próximos das médias do processo
- **20% cases-gargalo**: alta complexidade e alta prioridade, simulando picos sazonais (BPI Challenge 2017 reporta 18-22% de casos em fast-track)
- **10% cases-exceção**: caminhos atípicos, baixa probabilidade de sucesso, possíveis falhas e reaberturas

Cada case carrega cinco atributos numéricos que compõem o vetor `case_context` consumido pelo agente PPO (concatenado à representação GAT do workflow):

[TABELA 5.4 — Atributos sintéticos por case]

| Atributo | Domínio | Distribuição (cases normais) |
|---|---|---|
| priority | [0, 1] | Uniforme(0.3, 0.7) |
| value | R$ | LogNormal(μ=8.0, σ=0.5) |
| complexity | [0.5, 3.0] | Normal(1.0, 0.1) clipped |
| customer_tier | [0, 1] | Uniforme(0.3, 0.8) |
| urgency | [0, 1] | Uniforme(0.2, 0.6) |

Fonte: O autor (2026), calibrado conforme distribuições observadas em van der Aalst et al. (2011).

Atributos textuais (nome de cliente, ID único) foram gerados via biblioteca Faker (locale `pt_BR`), apenas para realismo descritivo — não influenciam o aprendizado do agente.

### 5.3.2 Implementação

A geração foi implementada em `src/case_generator.py`. A função `generate_cases(n, process_name, seed)` produz a lista de Cases reprodutível: a mesma semente sempre gera o mesmo dataset. A distribuição é estritamente respeitada (resultado verificado em smoke test: 70.0% / 20.0% / 10.0% sobre 2.000 cases).

### 5.3.3 Validação via Process Mining

A representatividade dos datasets gerados foi validada via descoberta de processo seguida de conformance checking, seguindo o protocolo de van der Aalst (2016). Para cada XES, descobriu-se um modelo via Inductive Miner do PM4Py e calcularam-se quatro métricas de qualidade contra o BPMN original:

[TABELA 5.5 — Métricas de qualidade dos datasets sintéticos (meta: fitness ≥ 0.92)]

| Processo | Fitness | Precision | Generalization | Simplicity |
|---|---|---|---|---|
| A | {{FIT_A}} | {{PREC_A}} | {{GEN_A}} | {{SIMP_A}} |
| B | {{FIT_B}} | {{PREC_B}} | {{GEN_B}} | {{SIMP_B}} |
| C | {{FIT_C}} | {{PREC_C}} | {{GEN_C}} | {{SIMP_C}} |

Fonte: O autor (2026), métricas computadas via PM4Py 2.7.11.

> [NOTA: O texto a seguir deve ser ajustado conforme os valores obtidos. Se todos ≥ 0.92, declarar "todos os datasets atendem ao critério mínimo". Se algum abaixo, discutir as razões — provavelmente exceções no Processo C levam a fitness mais baixo.]

Os valores de fitness obtidos {{ATENDEM|NÃO ATENDEM}} ao critério estabelecido em §5.3, validando que os event logs sintéticos preservam a semântica estrutural dos BPMN originais. {{DISCUSSAO_OUTLIERS}}

## 5.4 Procedimento Experimental

### 5.4.1 Visão Geral do Pipeline

O pipeline experimental é composto por cinco estágios sequenciais, encadeados pelo script `src/train.py`:

1. Carregamento do BPMN (§4.3.1) — `bpmn_loader.load_bpmn()`
2. Geração ou carregamento dos cases (§5.3) — `case_generator.generate_cases()`
3. Execução das políticas baseline (FIFO, prioridade-fixa) para extração das métricas de referência da função de recompensa
4. Treinamento PPO+GAT por `total_timesteps` em `n_envs` ambientes paralelos (SubprocVecEnv), com avaliação a cada 25 mil timesteps
5. Persistência de checkpoints, modelo final e métricas no MLflow

Cada uma das três sementes (42, 123, 7) gera um run MLflow independente, totalizando nove runs por processo × processo = 9 runs principais. Cada run consome aproximadamente {{TEMPO_RUN}} de tempo de relógio em {{HARDWARE_TREINO}}.

### 5.4.2 Resultados dos Baselines

[NOTA: Os baselines servem dupla função: (a) parametrizar a função de recompensa multi-objetivo via BaselineMetrics; (b) ponto de comparação direto para H5. Esta seção apresenta os valores obtidos pelo FIFO e pela prioridade-fixa antes do treinamento PPO+GAT.]

[TABELA 5.6 — Métricas das políticas baseline (médias sobre 2.000 cases, 3 seeds)]

| Política | Processo | Cycle time (s) | \|V\| executado | \|E\| executado | Taxa de sucesso |
|---|---|---|---|---|---|
| FIFO | A | {{F_CT_A}} | {{F_V_A}} | {{F_E_A}} | {{F_SR_A}} |
| FIFO | B | {{F_CT_B}} | {{F_V_B}} | {{F_E_B}} | {{F_SR_B}} |
| FIFO | C | {{F_CT_C}} | {{F_V_C}} | {{F_E_C}} | {{F_SR_C}} |
| Prioridade | A | {{P_CT_A}} | {{P_V_A}} | {{P_E_A}} | {{P_SR_A}} |
| Prioridade | B | {{P_CT_B}} | {{P_V_B}} | {{P_E_B}} | {{P_SR_B}} |
| Prioridade | C | {{P_CT_C}} | {{P_V_C}} | {{P_E_C}} | {{P_SR_C}} |

Fonte: O autor (2026).

A política de prioridade-fixa supera o FIFO em {{DELTA_PRIO_PCT}}% no tempo de ciclo médio dos três processos, confirmando que a modelagem manual madura (com pesos de branch ajustados por especialistas) extrai ganhos não-triviais frente ao roteamento ingênuo. Esta superioridade estabelece a "barreira competitiva" que o PPO+GAT precisa ultrapassar para que se justifique a complexidade adicional do aprendizado por reforço.

### 5.4.3 Treinamento PPO+GAT

[NOTA: O texto abaixo descreve o que VOCÊ FEZ. Não escreva isso até ter rodado de fato — os números precisam ser reais.]

O agente PPO+GAT foi treinado por {{TIMESTEPS_TOTAL}} timesteps em cada combinação processo × seed, totalizando {{TIMESTEPS_GRAND_TOTAL}} timesteps efetivos. A convergência foi monitorada em tempo real via MLflow e TensorBoard, com critério de parada antecipada caso o mean episode reward em janela de 100 episódios não apresentasse melhora superior a 0.01 por 500 mil timesteps consecutivos — critério {{ATIVADO|NÃO ATIVADO}} em {{N_RUNS_PARADAS}} das nove runs.

[FIGURA 5.4 — Curvas de mean episode reward por seed, agrupadas por processo]

A Figura 5.4 mostra a convergência das curvas de reward agregadas por processo (mediana ± IQR entre três seeds). Os três processos atingiram platô estável em {{TIMESTEP_PLATEAU}} timesteps, indicando que o orçamento de {{TIMESTEPS_TOTAL}} timesteps foi {{SUFICIENTE|INSUFICIENTE}}.

### 5.4.4 Experimento de Transferência (Hipótese H4)

Para avaliar empiricamente a Hipótese H4 (capacidade de generalização da representação aprendida), o agente treinado sobre o Processo B foi avaliado, **sem qualquer fine-tuning**, sobre o ambiente do Processo C. Foram executados 500 episódios determinísticos (deterministic=True na avaliação MaskablePPO).

A métrica de transferência adotada é a razão entre o reward médio do agente B aplicado a C e o reward médio do agente nativo de C:

```
transfer_ratio = E[R | π_B, env_C] / E[R | π_C, env_C]
```

Valor obtido: {{TRANSFER_RATIO}} (IC 95%: [{{TR_LOW}}, {{TR_HIGH}}]).

> [NOTA: H4 será considerada confirmada se transfer_ratio > 0.7 — ou seja, se o agente treinado em outro processo recupera pelo menos 70% do desempenho do agente especializado. Discutir o resultado obtido.]

### 5.4.5 Stress Test de Latência (Hipótese H3)

A API de inferência foi implementada via FastAPI (script `src/api_server.py`) e exposta em ambiente local. O stress test foi conduzido com Locust em quatro níveis de concorrência: 1, 10, 50 e 100 requisições simultâneas durante 5 minutos cada, totalizando ~{{TOTAL_REQS}} requisições por nível.

[TABELA 5.7 — Latência de inferência sob diferentes níveis de concorrência]

| Concorrência | Média (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Taxa de erro (%) |
|---|---|---|---|---|---|
| 1 | {{L_M_1}} | {{L_50_1}} | {{L_95_1}} | {{L_99_1}} | {{L_ERR_1}} |
| 10 | {{L_M_10}} | {{L_50_10}} | {{L_95_10}} | {{L_99_10}} | {{L_ERR_10}} |
| 50 | {{L_M_50}} | {{L_50_50}} | {{L_95_50}} | {{L_99_50}} | {{L_ERR_50}} |
| 100 | {{L_M_100}} | {{L_50_100}} | {{L_95_100}} | {{L_99_100}} | {{L_ERR_100}} |

Fonte: O autor (2026).

A latência p95 de {{L_95_100}} ms em concorrência 100 {{ATENDE|NÃO ATENDE}} o requisito não-funcional declarado em §1.5 (≤ 100 ms). H3 é, portanto, {{CONFIRMADA|REFUTADA}}.

## 5.5 Reposicionamento: do TCC 2018 ao Atual

[NOTA: Aqui vai o conteúdo que hoje é §5.1 do TCC — o "Reposicionamento do TCC 2018". Movido para o fim do capítulo porque (i) o orientador queria "o seu case" como protagonista, não o trabalho anterior, e (ii) faz mais sentido cronológico/narrativo encerrar contrastando "de onde se partiu" com "onde se chegou". O texto existente sobre TPS+SVM, o data lake, as 600 atividades, etc., entra aqui sem grandes mudanças — apenas reescrever o parágrafo de transição para refletir que agora aparece DEPOIS dos resultados, não antes.]

[TABELA 5.8 — Evolução entre o TCC 2018 e o presente trabalho]

[NOTA: Você já tem essa tabela (Tabela 2 do documento atual). Reaproveitar e atualizar se necessário.]

---

# CHECKLIST DE PREENCHIMENTO

Antes de submeter o capítulo ao orientador, verifique:

- [ ] Todos os `{{PLACEHOLDERS}}` substituídos por valores reais ou removidos com justificativa
- [ ] Todas as `[FIGURAS]` inseridas com legenda ABNT e referência cruzada
- [ ] Todas as `[TABELAS]` com fonte "O autor (2026)" abaixo
- [ ] Todas as `[NOTAS]` removidas
- [ ] Numeração de seções/tabelas/figuras consistente com o resto do TCC
- [ ] Referências bibliográficas adicionadas à lista final (Yin, van der Aalst, Schulman, Veličković, Brody, Dijkman)
- [ ] Discussão das hipóteses (H1-H5) feita em §6, não em §5 (§5 é setup + resultados de baseline; §6 é análise + comparação com trabalhos relacionados + decisão sobre hipóteses)
