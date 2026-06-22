"""Ambiente Gymnasium para simulação de execução de workflows BPMN.

Implementa o estágio 4.3.4 do TCC: ambiente compatível com a interface padrão
do Gymnasium, no qual cada episódio corresponde à execução completa de um
case sintético sobre um workflow BPMN.

Semântica de execução (token único, simplificada):
  - Eventos start: auto-avança para o próximo nó.
  - Tasks, intermediate, subprocess: auto-executam (acumulam tempo + custo)
    e avançam pelo único fluxo de saída.
  - Gateways exclusivos (xor_gw), inclusivos (inc_gw): pontos de decisão —
    o agente escolhe um dos fluxos de saída.
  - Gateways paralelos (par_gw): tratados como decisão exclusiva nesta versão
    do simulador (limitação documentada — token-passing concorrente fica como
    trabalho futuro, conforme §7.3 do TCC).
  - Eventos end: terminam o episódio.

Espaço de observação (Dict, com padding para tamanho fixo):
    x              — [MAX_N, NODE_FEATS] features de nó (com is_current/visited)
    edge_index     — [2, MAX_E] arestas (com -1 em posições inválidas)
    edge_attr      — [MAX_E, 1] probabilidades de branch
    node_mask      — [MAX_N] bool — True onde o nó é válido
    edge_mask      — [MAX_E] bool — True onde a aresta é válida
    current_node   — [MAX_N] one-hot do nó atual
    case_context   — [CASE_DIM] atributos do case corrente

Espaço de ação: Discrete(MAX_BRANCHES). Em cada gateway, mascarado para os
fluxos de saída efetivamente disponíveis. Compatível com sb3-contrib
MaskablePPO via método `action_masks()`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .bpmn_loader import BPMNGraph, NODE_FEATS
from .case_generator import Case
from .reward import MultiObjectiveReward

logger = logging.getLogger(__name__)

# Constantes de padding — dimensionadas para acomodar os três processos
# representativos do TCC (Processo B é o maior, com 22 nós e 31 arestas).
MAX_NODES = 64
MAX_EDGES = 128
MAX_BRANCHES = 8
CASE_DIM = 5


class WorkflowEnv(gym.Env):
    """Ambiente single-token para um workflow BPMN sobre cases sintéticos."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        graph: BPMNGraph,
        cases: List[Case],
        reward_fn: MultiObjectiveReward,
        max_steps: int = 200,
        max_nodes: int = MAX_NODES,
        max_edges: int = MAX_EDGES,
        max_branches: int = MAX_BRANCHES,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if graph.num_nodes > max_nodes:
            raise ValueError(
                f"Grafo tem {graph.num_nodes} nós; aumente MAX_NODES (atual {max_nodes})"
            )
        if graph.num_edges > max_edges:
            raise ValueError(
                f"Grafo tem {graph.num_edges} arestas; aumente MAX_EDGES (atual {max_edges})"
            )

        self.graph = graph
        self.cases = cases
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_branches = max_branches

        # Pré-computa tensores estáticos do grafo (não mudam durante o treino)
        data = graph.to_torch_geometric()
        self._x_static = np.zeros((max_nodes, NODE_FEATS), dtype=np.float32)
        self._x_static[: graph.num_nodes] = data.x.numpy()

        self._edge_index_static = np.full((2, max_edges), -1, dtype=np.int64)
        self._edge_index_static[:, : graph.num_edges] = data.edge_index.numpy()

        self._edge_attr_static = np.zeros((max_edges, 1), dtype=np.float32)
        self._edge_attr_static[: graph.num_edges] = data.edge_attr.numpy()

        self._node_mask = np.zeros(max_nodes, dtype=bool)
        self._node_mask[: graph.num_nodes] = True

        self._edge_mask = np.zeros(max_edges, dtype=bool)
        self._edge_mask[: graph.num_edges] = True

        # Espaços Gymnasium
        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=0.0, high=1.0,
                            shape=(max_nodes, NODE_FEATS), dtype=np.float32),
            "edge_index": spaces.Box(low=-1, high=max_nodes,
                                      shape=(2, max_edges), dtype=np.int64),
            "edge_attr": spaces.Box(low=0.0, high=1.0,
                                     shape=(max_edges, 1), dtype=np.float32),
            "node_mask": spaces.Box(low=0, high=1,
                                     shape=(max_nodes,), dtype=bool),
            "edge_mask": spaces.Box(low=0, high=1,
                                     shape=(max_edges,), dtype=bool),
            "current_node": spaces.Box(low=0.0, high=1.0,
                                        shape=(max_nodes,), dtype=np.float32),
            "case_context": spaces.Box(low=-3.0, high=3.0,
                                        shape=(CASE_DIM,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(max_branches)

        self._rng = np.random.default_rng(seed)
        self._reset_state()

    # ------------------------------------------------------------------
    # Estado interno
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._current_node = self.graph.start_idx
        self._visited_nodes: set[int] = set()
        self._executed_edges: set[int] = set()
        self._elapsed_time = 0.0
        self._total_cost = 0.0
        self._quality_sum = 0.0
        self._quality_count = 0
        self._steps = 0
        self._success = False
        self._available_branches: List[int] = []
        self._current_mask = np.zeros(self.max_branches, dtype=bool)
        self._case: Optional[Case] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Seleciona case (determinístico se options["case_idx"] for fornecido)
        case_idx = options.get("case_idx") if options else None
        if case_idx is None:
            case_idx = int(self._rng.integers(0, len(self.cases)))
        else:
            case_idx = int(case_idx) % len(self.cases)
        self._case = self.cases[case_idx]

        self._reset_state()
        self._case = self.cases[case_idx]

        # Auto-avança até primeiro ponto de decisão (ou end direto)
        self._auto_advance()
        return self._obs(), self._info()

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self._steps += 1
        truncated = self._steps >= self.max_steps
        terminated = False
        reward = 0.0
        info_extra: Dict[str, Any] = {}

        # Caso já estejamos em um end event, encerra imediatamente
        if self.graph.nodes[self._current_node].type == "end":
            terminated = True
            self._success = self._compute_success()
            reward += self.reward_fn.terminal(
                elapsed_time=self._elapsed_time,
                nodes_executed=len(self._visited_nodes),
                edges_executed=len(self._executed_edges),
                avg_quality=self._avg_quality(),
                success=self._success,
            )
            return self._obs(), reward, terminated, truncated, self._info(info_extra)

        action = int(action)
        # Defesa contra ação fora do espaço (não deve ocorrer com MaskablePPO,
        # mas se ocorrer aplica penalidade e termina)
        if (
            action < 0
            or action >= self.max_branches
            or action >= len(self._available_branches)
            or not self._current_mask[action]
        ):
            info_extra["invalid_action"] = True
            terminated = True
            self._success = False
            reward = -1.0  # penalidade dura
            return self._obs(), reward, terminated, truncated, self._info(info_extra)

        # Aplica escolha da branch
        chosen_edge_idx = self._available_branches[action]
        _, target, _ = self.graph.edges[chosen_edge_idx]
        self._executed_edges.add(chosen_edge_idx)
        self._current_node = target

        # Contabiliza qualidade da branch escolhida
        q = self._branch_quality(chosen_edge_idx)
        self._quality_sum += q
        self._quality_count += 1

        # Custo de step (incentiva episódios curtos)
        reward += -0.01

        # Auto-avança pelas tarefas determinísticas até próximo ponto de decisão
        self._auto_advance()

        # Verifica terminal
        if self.graph.nodes[self._current_node].type == "end":
            terminated = True
            self._success = self._compute_success()
            reward += self.reward_fn.terminal(
                elapsed_time=self._elapsed_time,
                nodes_executed=len(self._visited_nodes),
                edges_executed=len(self._executed_edges),
                avg_quality=self._avg_quality(),
                success=self._success,
            )
        elif not self._available_branches:
            # Beco sem saída (BPMN malformado ou loop infinito truncado)
            terminated = True
            self._success = False
            reward += self.reward_fn.terminal(
                elapsed_time=self._elapsed_time,
                nodes_executed=len(self._visited_nodes),
                edges_executed=len(self._executed_edges),
                avg_quality=self._avg_quality(),
                success=False,
            )

        return self._obs(), reward, terminated, truncated, self._info(info_extra)

    # ------------------------------------------------------------------
    # Suporte para sb3-contrib MaskablePPO
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Máscara booleana sobre Discrete(max_branches)."""
        return self._current_mask.copy()

    # ------------------------------------------------------------------
    # Lógica de simulação
    # ------------------------------------------------------------------

    def _auto_advance(self) -> None:
        """Avança até um ponto de decisão, end event, ou beco sem saída."""
        # Guard contra loops infinitos em BPMN cíclicos (ex.: Processo C)
        guard = 0
        max_guard = self.max_nodes * 4

        while guard < max_guard:
            guard += 1
            self._visited_nodes.add(self._current_node)
            ntype = self.graph.nodes[self._current_node].type

            if ntype == "end":
                self._available_branches = []
                self._current_mask[:] = False
                return

            successors = self.graph.successors(self._current_node)
            outgoing_idxs = self.graph.outgoing_edge_indices(self._current_node)

            if ntype in ("task", "intermediate", "subprocess", "start"):
                # Auto-executa: acumula tempo/custo (exceto start, que é "instantâneo")
                if ntype != "start":
                    node = self.graph.nodes[self._current_node]
                    mult = self._case.duration_multiplier if self._case else 1.0
                    noise = 1.0 + 0.2 * float(self._rng.normal())
                    self._elapsed_time += max(node.duration_s * mult * noise, 1.0)
                    self._total_cost += node.cost

                if not successors:
                    # Dead-end sem ser end event
                    self._available_branches = []
                    self._current_mask[:] = False
                    return
                if len(successors) == 1:
                    self._executed_edges.add(outgoing_idxs[0])
                    self._current_node = successors[0][0]
                    continue
                # Múltiplos sucessores em task é incomum: trata como decisão
                self._set_decision_branches(outgoing_idxs)
                return

            if ntype in ("xor_gw", "par_gw", "inc_gw"):
                if not successors:
                    self._available_branches = []
                    self._current_mask[:] = False
                    return
                if len(successors) == 1:
                    # Gateway com 1 saída — auto-toma
                    self._executed_edges.add(outgoing_idxs[0])
                    self._current_node = successors[0][0]
                    continue
                # Ponto de decisão genuíno
                self._set_decision_branches(outgoing_idxs)
                return

            # Tipo desconhecido — abortar
            logger.warning("Tipo de nó não tratado: %s", ntype)
            return

    def _set_decision_branches(self, outgoing_idxs: List[int]) -> None:
        """Configura as branches disponíveis e a máscara de ação."""
        # Limita ao máximo configurado (mais branches que isso é raro em BPMN real)
        self._available_branches = list(outgoing_idxs[: self.max_branches])
        self._current_mask = np.zeros(self.max_branches, dtype=bool)
        for k in range(len(self._available_branches)):
            self._current_mask[k] = True

    def _branch_quality(self, edge_idx: int) -> float:
        """Qualidade da branch escolhida ∈ [0,1].

        Heurística sintética: cases com alta prioridade preferem branches
        de alta probabilidade (caminhos "bem trilhados"); cases de baixa
        prioridade toleram desvios. O agente deve aprender essa correlação
        a partir do `case_context` na observação.
        """
        _, _, prob = self.graph.edges[edge_idx]
        if self._case is None:
            return 0.5
        ideal_prob = 0.5 + 0.3 * (self._case.priority - 0.5)
        return float(np.clip(1.0 - abs(prob - ideal_prob), 0.0, 1.0))

    def _avg_quality(self) -> float:
        if self._quality_count == 0:
            return 0.5  # neutro
        return self._quality_sum / self._quality_count

    def _compute_success(self) -> bool:
        """Sucesso se a qualidade média das decisões > 0.5 E chegou a end event."""
        ended_well = self.graph.nodes[self._current_node].type == "end"
        return ended_well and self._avg_quality() > 0.5

    # ------------------------------------------------------------------
    # Construção de observação / info
    # ------------------------------------------------------------------

    def _obs(self) -> Dict[str, np.ndarray]:
        x = self._x_static.copy()
        # Zera flags dinâmicas e reaplica
        x[:, NODE_FEATS - 2] = 0.0  # is_current
        x[:, NODE_FEATS - 1] = 0.0  # visited
        for n in self._visited_nodes:
            if 0 <= n < self.max_nodes:
                x[n, NODE_FEATS - 1] = 1.0
        if 0 <= self._current_node < self.max_nodes:
            x[self._current_node, NODE_FEATS - 2] = 1.0

        current_oh = np.zeros(self.max_nodes, dtype=np.float32)
        if 0 <= self._current_node < self.max_nodes:
            current_oh[self._current_node] = 1.0

        case_vec = self._case.to_vector() if self._case else np.zeros(CASE_DIM, dtype=np.float32)

        return {
            "x": x,
            "edge_index": self._edge_index_static,
            "edge_attr": self._edge_attr_static,
            "node_mask": self._node_mask,
            "edge_mask": self._edge_mask,
            "current_node": current_oh,
            "case_context": case_vec,
        }

    def _info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info = {
            "action_mask": self._current_mask.copy(),
            "elapsed_time": float(self._elapsed_time),
            "total_cost": float(self._total_cost),
            "nodes_executed": len(self._visited_nodes),
            "edges_executed": len(self._executed_edges),
            "success": bool(self._success),
            "avg_quality": float(self._avg_quality()),
            "case_id": self._case.case_id if self._case else "",
            "case_category": self._case.category if self._case else "",
        }
        if extra:
            info.update(extra)
        return info
