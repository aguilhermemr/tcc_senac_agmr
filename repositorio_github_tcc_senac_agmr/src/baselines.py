"""Políticas baseline para comparação com PPO+GAT.

Implementa as duas linhas-base mencionadas em §6.2.6 do TCC:

  - FIFO: no encontro de cada gateway, escolhe a primeira branch válida
    (ordem do arquivo BPMN). Representa o comportamento default dos motores
    BPM tradicionais sem inteligência.

  - Prioridade-fixa: escolhe a branch com maior probabilidade prior
    (atributo `probability` nas sequenceFlow extensions). Representa o
    "modelagem manual madura por especialistas".

Ambas servem para computar `BaselineMetrics` que parametrizam a função de
recompensa multi-objetivo (R_tempo, R_complexidade referem-se a esse baseline)
E como ponto de comparação direto na hipótese H5.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .bpmn_loader import BPMNGraph
from .case_generator import Case
from .reward import BaselineMetrics, MultiObjectiveReward, RewardConfig
from .workflow_env import WorkflowEnv


PolicyFn = Callable[[Dict[str, np.ndarray], Dict], int]


def fifo_policy(obs: Dict[str, np.ndarray], info: Dict) -> int:
    """Sempre escolhe a primeira branch válida."""
    mask = info.get("action_mask")
    if mask is None:
        return 0
    valid = np.where(mask)[0]
    return int(valid[0]) if len(valid) else 0


def make_priority_policy(env: WorkflowEnv) -> PolicyFn:
    """Cria política que escolhe a branch de maior probabilidade prior."""
    def _policy(obs: Dict[str, np.ndarray], info: Dict) -> int:
        mask = info.get("action_mask")
        if mask is None:
            return 0
        valid_actions = np.where(mask)[0]
        if not len(valid_actions):
            return 0
        # Pega a probabilidade prior de cada branch disponível
        probs = []
        for a in valid_actions:
            if a < len(env._available_branches):
                edge_idx = env._available_branches[a]
                probs.append(env.graph.edges[edge_idx][2])
            else:
                probs.append(0.0)
        return int(valid_actions[int(np.argmax(probs))])
    return _policy


def _run_episodes(
    env: WorkflowEnv,
    policy_fn: PolicyFn,
    n_episodes: int,
) -> Dict[str, float]:
    """Executa n_episodes na env coletando estatísticas agregadas."""
    cycle_times: List[float] = []
    nodes: List[int] = []
    edges: List[int] = []
    successes: List[bool] = []

    for i in range(n_episodes):
        obs, info = env.reset(options={"case_idx": i % len(env.cases)})
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = policy_fn(obs, info)
            obs, _reward, terminated, truncated, info = env.step(action)
        cycle_times.append(info["elapsed_time"])
        nodes.append(info["nodes_executed"])
        edges.append(info["edges_executed"])
        successes.append(info["success"])

    cycle_mean = float(np.mean(cycle_times)) if cycle_times else 1.0
    return {
        "cycle_time": cycle_mean,
        "nodes": int(round(float(np.mean(nodes)))) if nodes else 1,
        "edges": int(round(float(np.mean(edges)))) if edges else 1,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "throughput": 3600.0 / max(cycle_mean, 1.0)  # cases/hora
                       * (float(np.mean(successes)) if successes else 0.0),
    }


def run_baseline_fifo(
    graph: BPMNGraph,
    cases: List[Case],
    seed: int = 0,
) -> BaselineMetrics:
    """Computa BaselineMetrics rodando a política FIFO sobre os cases."""
    dummy_baseline = BaselineMetrics(cycle_time=1.0, nodes=1, edges=1, throughput=1.0)
    dummy_reward = MultiObjectiveReward(RewardConfig(), dummy_baseline)
    env = WorkflowEnv(graph, cases, dummy_reward, seed=seed)
    stats = _run_episodes(env, fifo_policy, n_episodes=len(cases))
    return BaselineMetrics(
        cycle_time=stats["cycle_time"],
        nodes=max(stats["nodes"], 1),
        edges=max(stats["edges"], 1),
        throughput=stats["throughput"],
    )


def run_baseline_priority(
    graph: BPMNGraph,
    cases: List[Case],
    seed: int = 0,
) -> Dict[str, float]:
    """Roda a política de prioridade-fixa e retorna estatísticas agregadas."""
    dummy_baseline = BaselineMetrics(cycle_time=1.0, nodes=1, edges=1, throughput=1.0)
    dummy_reward = MultiObjectiveReward(RewardConfig(), dummy_baseline)
    env = WorkflowEnv(graph, cases, dummy_reward, seed=seed)
    policy = make_priority_policy(env)
    return _run_episodes(env, policy, n_episodes=len(cases))
