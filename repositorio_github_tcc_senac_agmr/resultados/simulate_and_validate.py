"""Gera event logs reais para os 3 processos BPMN e valida via PM4Py.

Reproduz a metodologia do §5.3 do TCC sem depender de torch/torch-geometric.
"""
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import random

from lxml import etree
import networkx as nx
import pandas as pd
import pm4py

NS = {"bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
      "zeebe": "http://camunda.org/schema/zeebe/1.0"}


def load_bpmn_graph(path):
    tree = etree.parse(str(path))
    process = tree.getroot().find("bpmn:process", NS)
    G = nx.DiGraph()
    durations = {}
    for elem in process:
        tag = etree.QName(elem.tag).localname
        eid = elem.get("id")
        if not eid: continue
        name = elem.get("name", eid)
        if "Event" in tag: ntype = "event"
        elif "Gateway" in tag: ntype = "gateway"
        elif "Task" in tag or "Activity" in tag or tag == "subProcess": ntype = "task"
        elif tag == "sequenceFlow": continue
        else: continue
        G.add_node(eid, name=name, type=ntype, tag=tag)
        dur = 0
        for prop in elem.findall(".//zeebe:property", NS):
            if prop.get("name") == "duration":
                try: dur = int(prop.get("value", "0"))
                except ValueError: dur = 0
        durations[eid] = dur
    for sf in process.findall("bpmn:sequenceFlow", NS):
        src, tgt = sf.get("sourceRef"), sf.get("targetRef")
        if src and tgt: G.add_edge(src, tgt)
    return G, durations


def find_start(G):
    for n in G.nodes:
        if G.in_degree(n) == 0 and G.nodes[n]["type"] == "event":
            return n
    return None


def simulate_case(G, durations, start, case_attrs, rng):
    events = []
    t = datetime(2026, 1, 1) + timedelta(seconds=case_attrs.get("case_id", 0) * 600)
    cur = start
    visited = defaultdict(int)
    mult = case_attrs.get("duration_multiplier", 1.0)
    cat = case_attrs.get("category", "normal")
    if cat == "bottleneck": mult *= rng.uniform(1.5, 3.0)
    elif cat == "exception": mult *= rng.uniform(0.8, 1.2)

    for _ in range(100):
        nd = G.nodes[cur]
        ntype, nname = nd["type"], nd["name"]
        if ntype == "task":
            events.append({"activity": nname, "timestamp": t})
            t += timedelta(seconds=int(durations.get(cur, 60) * mult))
        succ = list(G.successors(cur))
        if not succ:
            if ntype == "event": events.append({"activity": nname, "timestamp": t})
            break
        if len(succ) == 1:
            cur = succ[0]
        else:
            visited[cur] += 1
            if "Resolvido" in nname and cat == "exception": idx = 1 if len(succ) > 1 else 0
            elif "RetornoOuN2" in nname: idx = 0 if visited[cur] < 2 else 1
            else: idx = 0
            cur = succ[idx]
    return events


def main():
    BPMN = Path("/mnt/user-data/outputs/angelia/exemplos")
    DATA = Path("/tmp/datasets")
    OUT = Path("/tmp/event_logs"); OUT.mkdir(exist_ok=True)
    SEED = 42
    rng = random.Random(SEED)
    metrics = {}

    for L in ["A", "B", "C"]:
        print(f"\n=== Processo {L} ===")
        G, durs = load_bpmn_graph(BPMN / f"process_{L}.bpmn")
        start = find_start(G)
        cases = json.loads((DATA / f"process_{L}_seed{SEED}.json").read_text())
        rows = []
        for i, case in enumerate(cases):
            case["case_id"] = i
            for ev in simulate_case(G, durs, start, case, rng):
                rows.append({"case:concept:name": f"case_{i}",
                             "concept:name": ev["activity"],
                             "time:timestamp": ev["timestamp"],
                             "case:category": case["category"]})
        df = pd.DataFrame(rows)
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
        df = df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
        df.to_csv(OUT / f"process_{L}.csv", index=False)
        log = pm4py.format_dataframe(df, case_id="case:concept:name",
                                       activity_key="concept:name",
                                       timestamp_key="time:timestamp")
        net, im, fm = pm4py.discover_petri_net_inductive(log)
        try: fit = pm4py.fitness_token_based_replay(log, net, im, fm)
        except Exception: fit = None
        try: prec = float(pm4py.precision_token_based_replay(log, net, im, fm))
        except Exception: prec = None
        try: gen = float(pm4py.generalization_tbr(log, net, im, fm))
        except Exception: gen = None
        try: simp = float(pm4py.simplicity_petri_net(net, im, fm))
        except Exception: simp = None
        fit_val = fit.get("log_fitness", 0.0) if isinstance(fit, dict) else (float(fit) if fit else None)
        metrics[L] = {"n_cases": df["case:concept:name"].nunique(),
                      "n_events": len(df),
                      "n_unique_activities": df["concept:name"].nunique(),
                      "fitness": fit_val, "precision": prec,
                      "generalization": gen, "simplicity": simp}
        print(f"  Fitness={fit_val} Precision={prec} Gen={gen} Simp={simp}")

    (OUT / "pm4py_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\n=== Salvo em {OUT}/pm4py_metrics.json ===")


if __name__ == "__main__":
    main()
