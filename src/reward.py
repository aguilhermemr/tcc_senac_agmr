"""Função de recompensa multi-objetivo R(s, a).

Implementa fielmente §4.4 do TCC:

    R = w_T · R_tempo + w_C · R_complexidade + w_Q · R_qualidade + w_S · R_sucesso

Componentes (todos limitados a [-1, +1] antes da ponderação):

  R_tempo        — proporcional à redução do tempo de ciclo vs. baseline FIFO.
                   Positivo se o caso terminou mais rápido que a média baseline.
  R_complexidade — proporcional à redução de |V| + |E| executados vs. baseline.
                   Positivo se o agente "podou" caminhos.
  R_qualidade    — média das qualidades das branches escolhidas (no env),
                   reescalada de [0,1] para [-1,+1].
  R_sucesso      — recompensa terminal binária (+1 sucesso / -1 falha).

Pesos default (0.3, 0.3, 0.3, 0.1) calibrados via grid search inicial
conforme §4.4. R_sucesso recebe peso menor por sua esparsidade — mas como é
de magnitude alta (+1/-1), continua sendo decisiva na convergência.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaselineMetrics:
    """Métricas baseline (FIFO) usadas como referência para a recompensa."""
    cycle_time: float = 600.0   # segundos
    nodes: int = 10
    edges: int = 12
    throughput: float = 1.0     # cases/hora


@dataclass
class RewardConfig:
    """Pesos da função multi-objetivo. Devem somar ~1.0 por convenção."""
    w_time: float = 0.30
    w_complexity: float = 0.30
    w_quality: float = 0.30
    w_success: float = 0.10


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class MultiObjectiveReward:
    """Computa R(s, a) terminal a partir das métricas acumuladas no episódio."""

    def __init__(self, config: RewardConfig, baseline: BaselineMetrics):
        self.cfg = config
        self.baseline = baseline

    def terminal(
        self,
        elapsed_time: float,
        nodes_executed: int,
        edges_executed: int,
        avg_quality: float,
        success: bool,
    ) -> float:
        bt = max(self.baseline.cycle_time, 1.0)
        bn = max(self.baseline.nodes, 1)
        be = max(self.baseline.edges, 1)

        # R_tempo: 0 quando igual ao baseline; positivo quando mais rápido.
        r_time = _clip(-(elapsed_time / bt - 1.0))

        # R_complexidade: 0 quando igual; positivo quando usou menos nós+arestas.
        r_complex = _clip(1.0 - (nodes_executed + edges_executed) / (bn + be))

        # R_qualidade: [0,1] → [-1,+1]
        r_quality = _clip(2.0 * avg_quality - 1.0)

        # R_sucesso: binário terminal
        r_success = 1.0 if success else -1.0

        return (
            self.cfg.w_time * r_time
            + self.cfg.w_complexity * r_complex
            + self.cfg.w_quality * r_quality
            + self.cfg.w_success * r_success
        )
