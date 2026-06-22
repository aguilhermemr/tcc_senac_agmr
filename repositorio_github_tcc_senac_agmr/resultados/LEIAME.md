# Resultados da Validação via PM4Py

Conteúdo desta pasta:

- `process_A.csv`, `process_B.csv`, `process_C.csv` — registros de eventos
  (event logs) gerados pela simulação dos cases sintéticos (semente 42) sobre
  cada BPMN, no formato compatível com PM4Py (colunas `case:concept:name`,
  `concept:name`, `time:timestamp`, `case:category`).
- `pm4py_metrics.json` — métricas reais medidas via PM4Py 2.7.22:
  - Inductive Miner para descoberta do modelo de processo
  - Token-based replay para fitness e precision
  - Generalization e simplicity sobre rede de Petri
- `simulate_and_validate.py` — script reprodutível que gera os event logs e
  calcula as métricas. Para reproduzir:

      pip install pm4py networkx lxml pandas numpy faker
      python resultados/simulate_and_validate.py

  (assume que os datasets já foram gerados via `src/case_generator.py`)

## Métricas resumidas (semente 42, 2000 cases por processo)

| Processo                     | Fitness | Precision | Generalization | Simplicity |
| ---------------------------- | ------- | --------- | -------------- | ---------- |
| A — Seleção de Pessoal       | 1,000   | 1,000     | 0,978          | 1,000      |
| B — Compras Corporativas     | 1,000   | 1,000     | 0,978          | 1,000      |
| C — Gestão de Incidentes     | 1,000   | 0,807     | 0,973          | 0,867      |

Fitness igual a 1,000 nos três processos é o resultado esperado para datasets
sintéticos gerados sem ruído, conforme discutido na seção 5.3.3 do TCC.
