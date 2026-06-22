"""Entry point para treinamento do agente PPO+GAT.

Uso:
    # Smoke test rápido (~2 minutos em CPU)
    python -m src.train --config configuracoes/default.yaml --smoke

    # Treino completo (~12-18h em GPU para 2M timesteps)
    python -m src.train --config configuracoes/default.yaml

    # Trocar BPMN sem editar o config:
    python -m src.train --config configuracoes/default.yaml \\
        --bpmn exemplos/process_B.bpmn --output-dir runs/process_B

Outputs (em output_dir):
    cases.json                 — dataset sintético gerado (reprodutível por seed)
    checkpoints/               — checkpoints periódicos do modelo PPO
    best/                      — melhor modelo segundo MaskableEvalCallback
    eval/                      — logs de avaliação
    tb/                        — TensorBoard logs
    final_model.zip            — modelo final salvo
    summary.txt                — resumo textual do run

Tudo também é logado em MLflow (file:./mlruns por default). Para inspecionar:
    mlflow ui --backend-store-uri file:./mlruns
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import mlflow
import numpy as np
import torch
import yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .baselines import run_baseline_fifo, run_baseline_priority
from .bpmn_loader import BPMNGraph, load_bpmn
from .case_generator import Case, generate_cases, load_cases, save_cases
from .gat_policy import GATGraphExtractor
from .reward import MultiObjectiveReward, RewardConfig
from .workflow_env import WorkflowEnv

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Dados
    bpmn_path: str = "exemplos/process_A.bpmn"
    cases_path: Optional[str] = None
    n_cases: int = 2000
    process_name: str = "process_A"
    seed: int = 42

    # Env
    max_steps: int = 200

    # Recompensa (§4.4 do TCC)
    w_time: float = 0.30
    w_complexity: float = 0.30
    w_quality: float = 0.30
    w_success: float = 0.10

    # GAT (§4.3.2)
    hidden_dim: int = 64
    heads: int = 4
    num_gat_layers: int = 3
    features_dim: int = 128
    dropout: float = 0.2

    # PPO (§4.2.3.2)
    total_timesteps: int = 2_000_000
    n_envs: int = 4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Logging
    output_dir: str = "runs/default"
    mlflow_uri: Optional[str] = None
    mlflow_experiment: str = "angelia-ppo-gat"
    use_wandb: bool = False
    wandb_project: str = "angelia-ppo-gat"

    # Avaliação / checkpointing
    eval_freq: int = 25_000
    n_eval_episodes: int = 20
    checkpoint_freq: int = 100_000


def load_config(path: str) -> TrainConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # Filtra apenas chaves conhecidas (tolera config com chaves extras)
    known = {f.name for f in TrainConfig.__dataclass_fields__.values()}
    raw_filtered = {k: v for k, v in raw.items() if k in known}
    return TrainConfig(**raw_filtered)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Env factory
# ----------------------------------------------------------------------

def make_env_factory(
    graph: BPMNGraph,
    cases: List[Case],
    reward_fn: MultiObjectiveReward,
    max_steps: int,
    seed: int,
):
    """Retorna uma callable para SB3 VecEnv."""
    def _make():
        env = WorkflowEnv(
            graph=graph,
            cases=cases,
            reward_fn=reward_fn,
            max_steps=max_steps,
            seed=seed,
        )
        env = Monitor(env, info_keywords=("success", "nodes_executed",
                                          "edges_executed", "case_category"))
        return env
    return _make


# ----------------------------------------------------------------------
# Callback de logging para MLflow
# ----------------------------------------------------------------------

class MLflowLoggingCallback(BaseCallback):
    """Mirror dos valores do logger interno do SB3 para MLflow."""

    def __init__(self, log_every: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self._last_log = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log < self.log_every:
            return True
        self._last_log = self.num_timesteps
        for key, val in self.model.logger.name_to_value.items():
            if isinstance(val, (int, float)) and not np.isnan(val):
                mlflow.log_metric(
                    key.replace("/", "_"),
                    float(val),
                    step=self.num_timesteps,
                )
        return True


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina PPO+GAT sobre BPMN.")
    parser.add_argument("--config", type=str, default="configuracoes/default.yaml")
    parser.add_argument("--bpmn", type=str, default=None,
                         help="sobrescreve bpmn_path do config")
    parser.add_argument("--timesteps", type=int, default=None,
                         help="sobrescreve total_timesteps do config")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="sobrescreve output_dir do config")
    parser.add_argument("--seed", type=int, default=None,
                         help="sobrescreve seed do config")
    parser.add_argument("--smoke", action="store_true",
                         help="modo smoke-test: ~2 minutos, valida pipeline ponta a ponta")
    return parser.parse_args()


def apply_smoke_overrides(cfg: TrainConfig) -> None:
    cfg.total_timesteps = 5_000
    cfg.n_envs = 2
    cfg.n_steps = 256
    cfg.batch_size = 32
    cfg.n_cases = 100
    cfg.eval_freq = 2_000
    cfg.checkpoint_freq = 5_000
    cfg.n_eval_episodes = 5


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config)

    if args.bpmn:
        cfg.bpmn_path = args.bpmn
    if args.timesteps:
        cfg.total_timesteps = args.timesteps
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.smoke:
        apply_smoke_overrides(cfg)

    set_seeds(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- BPMN
    logger.info("Carregando BPMN: %s", cfg.bpmn_path)
    graph = load_bpmn(cfg.bpmn_path)
    logger.info("  Processo: %s | %d nós | %d arestas",
                 graph.name, graph.num_nodes, graph.num_edges)

    # ------------------------------------------------------------- cases
    if cfg.cases_path and Path(cfg.cases_path).exists():
        logger.info("Carregando cases existentes de %s", cfg.cases_path)
        cases = load_cases(cfg.cases_path)
    else:
        logger.info("Gerando %d cases sintéticos (seed=%d)", cfg.n_cases, cfg.seed)
        cases = generate_cases(cfg.n_cases, cfg.process_name, seed=cfg.seed)
        cases_out = output_dir / "cases.json"
        save_cases(cases, cases_out)
        logger.info("  Cases salvos em %s", cases_out)

    # ---------------------------------------------------------- baseline
    logger.info("Computando baseline FIFO sobre %d cases", min(200, len(cases)))
    baseline = run_baseline_fifo(graph, cases[:200], seed=cfg.seed)
    logger.info("  FIFO: cycle=%.1fs | |V|=%d | |E|=%d | thr=%.2f cases/h",
                 baseline.cycle_time, baseline.nodes,
                 baseline.edges, baseline.throughput)

    logger.info("Computando baseline prioridade-fixa")
    prio_stats = run_baseline_priority(graph, cases[:200], seed=cfg.seed)
    logger.info("  Priority: cycle=%.1fs | succ=%.2f | thr=%.2f cases/h",
                 prio_stats["cycle_time"], prio_stats["success_rate"],
                 prio_stats["throughput"])

    reward_cfg = RewardConfig(
        w_time=cfg.w_time,
        w_complexity=cfg.w_complexity,
        w_quality=cfg.w_quality,
        w_success=cfg.w_success,
    )
    reward_fn = MultiObjectiveReward(reward_cfg, baseline)

    # -------------------------------------------------------------- env
    factory = make_env_factory(graph, cases, reward_fn, cfg.max_steps, cfg.seed)
    if cfg.n_envs > 1:
        vec_env = SubprocVecEnv([factory for _ in range(cfg.n_envs)])
    else:
        vec_env = DummyVecEnv([factory])
    eval_env = DummyVecEnv([factory])

    # ----------------------------------------------------------- policy
    policy_kwargs = dict(
        features_extractor_class=GATGraphExtractor,
        features_extractor_kwargs=dict(
            features_dim=cfg.features_dim,
            hidden_dim=cfg.hidden_dim,
            heads=cfg.heads,
            num_layers=cfg.num_gat_layers,
            dropout=cfg.dropout,
        ),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        learning_rate=cfg.learning_rate,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        verbose=1,
        seed=cfg.seed,
        tensorboard_log=str(output_dir / "tb"),
    )

    # -------------------------------------------------------- callbacks
    callbacks: List[BaseCallback] = [
        CheckpointCallback(
            save_freq=max(cfg.checkpoint_freq // max(cfg.n_envs, 1), 1),
            save_path=str(output_dir / "checkpoints"),
            name_prefix="ppo_gat",
        ),
        MaskableEvalCallback(
            eval_env,
            best_model_save_path=str(output_dir / "best"),
            log_path=str(output_dir / "eval"),
            eval_freq=max(cfg.eval_freq // max(cfg.n_envs, 1), 1),
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
        ),
        MLflowLoggingCallback(log_every=cfg.n_steps),
    ]

    # ------------------------------------------------------------ MLflow
    if cfg.mlflow_uri:
        mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)
    run_name = f"{cfg.process_name}_seed{cfg.seed}"

    with mlflow.start_run(run_name=run_name) as run:
        # Hiperparâmetros
        scalar_params = {k: v for k, v in asdict(cfg).items()
                         if isinstance(v, (int, float, str, bool))}
        mlflow.log_params(scalar_params)

        # Métricas de baseline
        mlflow.log_metrics({
            "baseline_fifo_cycle_time": baseline.cycle_time,
            "baseline_fifo_nodes": baseline.nodes,
            "baseline_fifo_edges": baseline.edges,
            "baseline_fifo_throughput": baseline.throughput,
            "baseline_priority_cycle_time": prio_stats["cycle_time"],
            "baseline_priority_success_rate": prio_stats["success_rate"],
            "baseline_priority_throughput": prio_stats["throughput"],
            "graph_num_nodes": graph.num_nodes,
            "graph_num_edges": graph.num_edges,
        })

        # Opcional: W&B
        if cfg.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=cfg.wandb_project,
                    config=scalar_params,
                    name=run_name,
                    sync_tensorboard=True,
                )
            except ImportError:
                logger.warning("wandb não instalado; ignorando.")

        # Treino
        logger.info("Iniciando treino: %d timesteps em %d envs",
                     cfg.total_timesteps, cfg.n_envs)
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        # Salva artefatos
        final_path = output_dir / "final_model"
        model.save(str(final_path))
        mlflow.log_artifact(str(final_path) + ".zip")
        mlflow.log_artifact(str(output_dir / "cases.json"))

        # Resumo
        summary = (
            f"Run ID:        {run.info.run_id}\n"
            f"Process:       {cfg.process_name}\n"
            f"BPMN:          {cfg.bpmn_path}\n"
            f"Timesteps:     {cfg.total_timesteps}\n"
            f"Seed:          {cfg.seed}\n"
            f"\n"
            f"Baseline FIFO  cycle={baseline.cycle_time:.1f}s "
            f"|V|={baseline.nodes} |E|={baseline.edges}\n"
            f"Baseline Prio  cycle={prio_stats['cycle_time']:.1f}s "
            f"succ={prio_stats['success_rate']:.2f}\n"
        )
        summary_path = output_dir / "summary.txt"
        summary_path.write_text(summary, encoding="utf-8")
        mlflow.log_artifact(str(summary_path))

        logger.info("\n%s", summary)
        logger.info("Run finalizado. Artefatos: %s", output_dir)


if __name__ == "__main__":
    main()
