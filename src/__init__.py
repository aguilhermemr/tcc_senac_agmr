"""Pacote angelia: pipeline PPO+GAT para otimização de workflows BPMN.

Implementa o estudo de caso descrito no Capítulo 5 do TCC
"Otimização de Workflows Empresariais com Machine Learning"
(Morales Romero, 2026).

Módulos:
    bpmn_loader     — BPMN 2.0 XML → torch_geometric.Data
    case_generator  — geração de cases sintéticos (70/20/10)
    reward          — função de recompensa multi-objetivo
    workflow_env    — Gymnasium env (simulação token-passing)
    gat_policy      — extrator GAT para sb3-contrib MaskablePPO
    baselines       — políticas FIFO e prioridade-fixa
    train           — entry point de treinamento
"""

__version__ = "0.1.0"
