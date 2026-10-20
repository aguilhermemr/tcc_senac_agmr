"""Gerador de cases sintéticos para treinamento.

Implementa a estratégia descrita em §5.3 do TCC: gera N cases por processo
com distribuição controlada 70/20/10 entre cases normais, gargalos e exceções.
Cada case carrega atributos numéricos que influenciam a simulação do workflow.

Tools utilizadas: Faker (textuais), numpy (numéricos). Ambas mencionadas no TCC.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from faker import Faker
    _HAS_FAKER = True
except ImportError:
    _HAS_FAKER = False


@dataclass
class Case:
    """Um caso (instância) sintética de execução de um processo de negócio.

    Atributos numéricos alimentam o vetor de contexto consumido pelo agente
    PPO (via concatenação com o embedding GAT do workflow).
    """
    case_id: str
    priority: float           # 0..1
    value: float              # quantia financeira (normalizada na vetorização)
    complexity: float         # 0.5..3.0 — afeta duração das tarefas
    customer_tier: float      # 0..1
    urgency: float            # 0..1
    category: str = "normal"  # normal | bottleneck | exception
    duration_multiplier: float = 1.0
    customer_name: str = ""

    def to_vector(self) -> np.ndarray:
        """Vetor de 5 dimensões consumido pelo extrator GAT (campo case_context)."""
        return np.array([
            self.priority,
            min(self.value / 100_000.0, 1.0),
            (self.complexity - 1.0),       # ~[-0.5, +2.0]
            self.customer_tier,
            self.urgency,
        ], dtype=np.float32)


def _make_faker(seed: int) -> Optional[Faker]:
    if not _HAS_FAKER:
        return None
    fake = Faker("pt_BR")
    Faker.seed(seed)
    return fake


def _draw_case(
    rng: np.random.Generator,
    fake: Optional[Faker],
    process_name: str,
    category: str,
    idx: int,
) -> Case:
    """Amostra um case dado uma categoria (normal / bottleneck / exception).

    Distribuições calibradas para refletir frequências comuns em event logs
    reais (BPI Challenge, Helpdesk, Sepsis), conforme citado em §5.3.
    """
    if category == "normal":
        priority = float(rng.uniform(0.3, 0.7))
        value = float(rng.lognormal(mean=8.0, sigma=0.5))
        complexity = float(np.clip(rng.normal(1.0, 0.1), 0.7, 1.4))
        tier = float(rng.uniform(0.3, 0.8))
        urgency = float(rng.uniform(0.2, 0.6))
    elif category == "bottleneck":
        priority = float(rng.uniform(0.4, 0.9))
        value = float(rng.lognormal(mean=9.0, sigma=0.7))
        complexity = float(np.clip(rng.normal(1.5, 0.2), 1.0, 2.2))
        tier = float(rng.uniform(0.5, 1.0))
        urgency = float(rng.uniform(0.6, 1.0))
    elif category == "exception":
        priority = float(rng.uniform(0.0, 0.3))
        value = float(rng.lognormal(mean=7.0, sigma=1.0))
        complexity = float(np.clip(rng.normal(1.8, 0.4), 1.2, 3.0))
        tier = float(rng.uniform(0.0, 0.4))
        urgency = float(rng.uniform(0.0, 0.5))
    else:
        raise ValueError(f"Categoria desconhecida: {category}")

    customer = fake.company() if fake else f"customer_{idx:05d}"

    return Case(
        case_id=f"{process_name}_{category}_{idx:05d}",
        priority=priority,
        value=value,
        complexity=complexity,
        customer_tier=tier,
        urgency=urgency,
        category=category,
        duration_multiplier=complexity,
        customer_name=customer,
    )


def generate_cases(
    n: int,
    process_name: str,
    normal_frac: float = 0.70,
    bottleneck_frac: float = 0.20,
    exception_frac: float = 0.10,
    seed: int = 42,
) -> List[Case]:
    """Gera `n` cases com distribuição categorica controlada.

    Frações devem somar 1.0 (com tolerância de arredondamento). A semente
    é fixada para garantir reprodutibilidade — qualquer um pode regenerar
    o mesmo dataset a partir do código.
    """
    total = normal_frac + bottleneck_frac + exception_frac
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"Frações somam {total}, devem somar 1.0")

    rng = np.random.default_rng(seed)
    fake = _make_faker(seed)

    n_normal = int(round(n * normal_frac))
    n_bottleneck = int(round(n * bottleneck_frac))
    n_exception = n - n_normal - n_bottleneck

    cases: List[Case] = []
    for i in range(n_normal):
        cases.append(_draw_case(rng, fake, process_name, "normal", i))
    for i in range(n_bottleneck):
        cases.append(_draw_case(rng, fake, process_name, "bottleneck", i))
    for i in range(n_exception):
        cases.append(_draw_case(rng, fake, process_name, "exception", i))

    # Embaralha para não agrupar categorias na ordem de treino
    rng.shuffle(cases)
    return cases


def save_cases(cases: List[Case], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in cases], f, ensure_ascii=False, indent=2)


def load_cases(path: str | Path) -> List[Case]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Case(**d) for d in data]
