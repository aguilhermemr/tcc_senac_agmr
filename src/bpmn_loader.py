"""Carregador BPMN 2.0 → grafo PyTorch Geometric.

Implementa o estágio 4.3.1 da arquitetura proposta no TCC: conversão de um
arquivo BPMN 2.0 XML (exportado do Camunda Modeler) em uma representação
grafo-teórica processável pela rede GAT.

Elementos suportados (mapeados para 8 categorias one-hot):
    task         — userTask, serviceTask, scriptTask, manualTask, etc.
    xor_gw       — exclusiveGateway, eventBasedGateway
    par_gw       — parallelGateway
    inc_gw       — inclusiveGateway
    start        — startEvent
    end          — endEvent
    intermediate — intermediateThrow/CatchEvent, boundaryEvent
    subprocess   — subProcess, callActivity, transaction

Extensões opcionais (formato Camunda 8 / Zeebe):

    <bpmn:extensionElements>
      <zeebe:properties>
        <zeebe:property name="duration"    value="180"/>
        <zeebe:property name="cost"        value="50"/>
        <zeebe:property name="priority"    value="0.7"/>
        <zeebe:property name="critical"    value="true"/>
        <zeebe:property name="probability" value="0.6"/>  (em sequenceFlow)
      </zeebe:properties>
    </bpmn:extensionElements>

Quando ausentes, defaults são aplicados.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import torch
from lxml import etree
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# Namespaces BPMN 2.0 + extensões Camunda/Zeebe
NS = {
    "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
    "zeebe": "http://camunda.org/schema/zeebe/1.0",
    "camunda": "http://camunda.org/schema/1.0/bpmn",
}

# 8 categorias de nós (one-hot)
NODE_TYPES = [
    "task", "xor_gw", "par_gw", "inc_gw",
    "start", "end", "intermediate", "subprocess",
]
NODE_TYPE_IDX = {t: i for i, t in enumerate(NODE_TYPES)}
NUM_TYPES = len(NODE_TYPES)

# Layout das features de nó (14 dimensões — conforme §4.3.1 do TCC):
#   [0..7]   one-hot do tipo
#   [8]      duration_norm   (0..1, normalizado por duration_scale)
#   [9]      cost_norm       (0..1, normalizado por cost_scale)
#   [10]     priority        (0..1)
#   [11]     critical        (0/1)
#   [12]     is_current      (preenchido pelo env, 0/1)
#   [13]     visited         (preenchido pelo env, 0/1)
NODE_FEATS = NUM_TYPES + 6

# Mapa de tag XML → tipo canônico
ELEMENT_TO_TYPE: Dict[str, str] = {
    "task": "task",
    "userTask": "task",
    "serviceTask": "task",
    "scriptTask": "task",
    "manualTask": "task",
    "businessRuleTask": "task",
    "sendTask": "task",
    "receiveTask": "task",
    "exclusiveGateway": "xor_gw",
    "parallelGateway": "par_gw",
    "inclusiveGateway": "inc_gw",
    "eventBasedGateway": "xor_gw",  # comporta-se como escolha exclusiva
    "startEvent": "start",
    "endEvent": "end",
    "intermediateThrowEvent": "intermediate",
    "intermediateCatchEvent": "intermediate",
    "boundaryEvent": "intermediate",
    "subProcess": "subprocess",
    "callActivity": "subprocess",
    "transaction": "subprocess",
}


@dataclass
class BPMNNode:
    """Nó BPMN com atributos canônicos para simulação."""
    id: str
    name: str
    type: str
    duration_s: float = 60.0
    cost: float = 1.0
    priority: float = 0.5
    critical: bool = False


@dataclass
class BPMNGraph:
    """Grafo BPMN parseado, com auxiliares para conversão a PyG."""
    nodes: List[BPMNNode]
    edges: List[Tuple[int, int, float]]   # (src_idx, tgt_idx, branch_prob)
    id_to_idx: Dict[str, int]
    start_idx: int
    end_indices: List[int]
    name: str = ""

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def successors(self, idx: int) -> List[Tuple[int, float]]:
        """Retorna [(target_idx, branch_prob), ...] saindo de `idx`."""
        return [(t, p) for (s, t, p) in self.edges if s == idx]

    def outgoing_edge_indices(self, idx: int) -> List[int]:
        """Índices em self.edges que partem de `idx`."""
        return [i for i, (s, _, _) in enumerate(self.edges) if s == idx]

    def predecessors(self, idx: int) -> List[int]:
        return [s for (s, t, _) in self.edges if t == idx]

    def to_torch_geometric(
        self,
        duration_scale: float = 600.0,
        cost_scale: float = 100.0,
    ) -> Data:
        """Cria objeto PyG Data com features estáticas (sem flags dinâmicas)."""
        n = self.num_nodes
        x = torch.zeros((n, NODE_FEATS), dtype=torch.float32)
        for i, node in enumerate(self.nodes):
            x[i, NODE_TYPE_IDX[node.type]] = 1.0
            x[i, NUM_TYPES + 0] = min(node.duration_s / duration_scale, 1.0)
            x[i, NUM_TYPES + 1] = min(node.cost / cost_scale, 1.0)
            x[i, NUM_TYPES + 2] = float(node.priority)
            x[i, NUM_TYPES + 3] = 1.0 if node.critical else 0.0
            # is_current e visited preenchidos pelo env em runtime
        if self.edges:
            ei = torch.tensor(
                [[s for (s, _, _) in self.edges],
                 [t for (_, t, _) in self.edges]],
                dtype=torch.long,
            )
            ea = torch.tensor(
                [[p] for (_, _, p) in self.edges],
                dtype=torch.float32,
            )
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)
            ea = torch.zeros((0, 1), dtype=torch.float32)
        return Data(x=x, edge_index=ei, edge_attr=ea)


def _strip_ns(tag: str) -> str:
    """Remove o namespace de uma tag XML: '{...}foo' → 'foo'."""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _parse_extensions(elem) -> Dict[str, str]:
    """Lê extensionElements/{zeebe,camunda}:property em um dict name→value."""
    props: Dict[str, str] = {}
    for ext in elem.findall("bpmn:extensionElements", NS):
        for properties in ext.findall("zeebe:properties", NS):
            for prop in properties.findall("zeebe:property", NS):
                k = prop.get("name")
                v = prop.get("value")
                if k:
                    props[k] = v
        for prop in ext.findall("camunda:property", NS):
            k = prop.get("name")
            v = prop.get("value")
            if k:
                props.setdefault(k, v)
    return props


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_bpmn(
    path: str | Path,
    default_duration_s: float = 60.0,
    default_cost: float = 1.0,
) -> BPMNGraph:
    """Lê um .bpmn (BPMN 2.0 XML) e retorna o BPMNGraph correspondente.

    Levanta ValueError se o arquivo não contém <bpmn:process> ou não tem startEvent.
    """
    path = Path(path)
    tree = etree.parse(str(path))
    root = tree.getroot()

    processes = root.findall("bpmn:process", NS)
    if not processes:
        raise ValueError(f"Nenhum <bpmn:process> encontrado em {path}")
    if len(processes) > 1:
        logger.warning("%d processos em %s; usando o primeiro", len(processes), path)
    process = processes[0]
    process_name = process.get("name") or process.get("id") or path.stem

    nodes: List[BPMNNode] = []
    id_to_idx: Dict[str, int] = {}

    for child in process:
        tag = _strip_ns(child.tag)
        if tag not in ELEMENT_TO_TYPE:
            continue
        props = _parse_extensions(child)
        duration = _coerce_float(props.get("duration"), default_duration_s)
        cost = _coerce_float(props.get("cost"), default_cost)
        priority = max(0.0, min(1.0, _coerce_float(props.get("priority"), 0.5)))
        critical = str(props.get("critical", "")).lower() in ("true", "1", "yes")
        node = BPMNNode(
            id=child.get("id"),
            name=child.get("name", "") or child.get("id"),
            type=ELEMENT_TO_TYPE[tag],
            duration_s=duration,
            cost=cost,
            priority=priority,
            critical=critical,
        )
        id_to_idx[node.id] = len(nodes)
        nodes.append(node)

    # Sequence flows
    raw_edges: List[Tuple[int, int, float]] = []
    for sf in process.findall("bpmn:sequenceFlow", NS):
        src = sf.get("sourceRef")
        tgt = sf.get("targetRef")
        if src in id_to_idx and tgt in id_to_idx:
            props = _parse_extensions(sf)
            prob = _coerce_float(props.get("probability"), 0.5)
            raw_edges.append((id_to_idx[src], id_to_idx[tgt], prob))
        else:
            logger.warning("Fluxo %s → %s ignorado (endpoint desconhecido)", src, tgt)

    # Normaliza probabilidades por nó-fonte: irmãos somam 1
    by_src: Dict[int, List[int]] = defaultdict(list)
    for i, (s, _, _) in enumerate(raw_edges):
        by_src[s].append(i)
    edges = list(raw_edges)
    for s, edge_ids in by_src.items():
        if len(edge_ids) > 1:
            total = sum(edges[i][2] for i in edge_ids) or 1.0
            for i in edge_ids:
                src, tgt, p = edges[i]
                edges[i] = (src, tgt, p / total)

    start_indices = [i for i, n in enumerate(nodes) if n.type == "start"]
    end_indices = [i for i, n in enumerate(nodes) if n.type == "end"]
    if not start_indices:
        raise ValueError(f"Nenhum startEvent em '{process_name}'")
    if not end_indices:
        logger.warning("Sem endEvent em '%s' — episódios truncarão por max_steps", process_name)

    return BPMNGraph(
        nodes=nodes,
        edges=edges,
        id_to_idx=id_to_idx,
        start_idx=start_indices[0],
        end_indices=end_indices,
        name=process_name,
    )


def to_networkx(g: BPMNGraph) -> nx.DiGraph:
    """Converte para networkx.DiGraph (útil para análises com PM4Py / inspeção)."""
    G = nx.DiGraph(name=g.name)
    for i, node in enumerate(g.nodes):
        G.add_node(i, **node.__dict__)
    for s, t, p in g.edges:
        G.add_edge(s, t, probability=p)
    return G
