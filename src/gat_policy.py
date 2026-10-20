"""Extrator de features baseado em GAT, compatível com sb3-contrib MaskablePPO.

Implementa o estágio 4.3.2 do TCC: codificação estrutural do workflow via uma
rede Graph Attention Network de 3 camadas com 4 heads de atenção, com pooling
mean+max produzindo um vetor de 128 dimensões que representa o workflow
inteiro, concatenado com o vetor de contexto do case.

Decisão de implementação:
    Usa-se GATv2Conv (Brody et al., 2022) em vez de GATConv original. GATv2 é
    estritamente mais expressivo que GAT v1 (provado no paper "How attentive
    are graph attention networks?"). Como o TCC menciona "Graph Attention
    Networks (GAT)" como categoria geral, GATv2 é uma escolha academicamente
    defensável e mais robusta na prática. Quem preferir fidelidade literal
    pode trocar GATv2Conv por GATConv: a interface é idêntica.

Decisão de arquitetura (variable-size graphs ↔ SB3):
    SB3 exige observações de tamanho fixo. PyTorch Geometric trabalha com
    grafos de tamanho variável. Solução adotada: padding até MAX_NODES e
    MAX_EDGES no env, e reconstrução para `Batch` PyG dentro do forward
    do extrator. O custo de reconstrução é da ordem de microssegundos por
    item (grafos têm ≤30 nós nos três processos do TCC), tornando-se
    negligenciável diante do custo das camadas GAT.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

from .bpmn_loader import NODE_FEATS


class GATGraphExtractor(BaseFeaturesExtractor):
    """Codifica observações grafo + case_context em um vetor de features fixo.

    Pipeline interno:
        1. Reconstrói o Batch PyG a partir do dict padded.
        2. Aplica 3 camadas GATv2 com 4 heads (ELU + dropout entre camadas).
        3. Pooling mean + max globais → 2*hidden_dim.
        4. Concatena com case_context.
        5. MLP de projeção → features_dim.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 128,
        hidden_dim: int = 64,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__(observation_space, features_dim)
        case_dim = observation_space["case_context"].shape[0]
        in_dim = NODE_FEATS

        # Camadas GAT empilhadas. Última camada com 1 head, sem concat,
        # para projetar para hidden_dim "puro".
        self.gat_layers = nn.ModuleList()
        for k in range(num_layers):
            is_last = k == num_layers - 1
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_dim if k == 0 else hidden_dim * heads,
                    out_channels=hidden_dim,
                    heads=1 if is_last else heads,
                    concat=not is_last,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )

        # Pooling produz 2*hidden_dim (mean + max). Concatena com case_context.
        pooled_dim = 2 * hidden_dim + case_dim
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
        )
        self.dropout_p = dropout

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = observations["x"]                      # [B, N, F]
        edge_index = observations["edge_index"]    # [B, 2, E]
        node_mask = observations["node_mask"].bool()  # [B, N]
        edge_mask = observations["edge_mask"].bool()  # [B, E]
        case_ctx = observations["case_context"]     # [B, C]

        batch_size = x.shape[0]
        device = x.device

        # Reconstrói cada grafo do batch como Data, depois Batch.from_data_list
        data_list = []
        for i in range(batch_size):
            nm = node_mask[i]
            em = edge_mask[i]
            xi = x[i][nm]                                   # [n_i, F]
            ei = edge_index[i][:, em].long()                # [2, e_i]
            # Defesa: descarta arestas com índices fora do range válido (padding leaks)
            n_valid = xi.shape[0]
            valid_e = (
                (ei[0] >= 0) & (ei[1] >= 0)
                & (ei[0] < n_valid) & (ei[1] < n_valid)
            )
            ei = ei[:, valid_e]
            data_list.append(Data(x=xi, edge_index=ei))

        batch = Batch.from_data_list(data_list).to(device)

        # Forward GAT
        h = batch.x
        for k, conv in enumerate(self.gat_layers):
            h = conv(h, batch.edge_index)
            if k < len(self.gat_layers) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout_p, training=self.training)

        # Pooling global por grafo do batch
        mean_p = global_mean_pool(h, batch.batch)   # [B, hidden_dim]
        max_p = global_max_pool(h, batch.batch)     # [B, hidden_dim]
        pooled = torch.cat([mean_p, max_p, case_ctx], dim=-1)

        return self.projection(pooled)
