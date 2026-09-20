"""Label Propagation based Source Identification (LPSI and LPSI_deg).

This module implements Algorithm 1 from Wang et al., "Multiple Source
Detection without Knowing the Underlying Propagation Model" (AAAI 2017).
The standard algorithm is defined on an undirected graph; directed task DAGs
are therefore converted to undirected graphs by default during localization.

``LPSI_deg`` is the degree-aware variant proposed by Hu et al. in Physica A
(2025).  It keeps the LPSI propagation and source-selection rules, but assigns
different positive initial labels to failed nodes according to node degree.
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import sparse


def lpsi_source_identification(
    graph: nx.Graph,
    alpha: float = 0.5,
    max_iter: int = 5,
    tol: float = 1e-8,
    mode: str = "iterative",
    source_count: Optional[int] = None,
    status_attr: str = "status",
    graph_mode: str = "undirected",
    degree_weighted: bool = False,
    degree_power: float = 1.0,
    degree_normalization: str = "none",
) -> Tuple[List[Hashable], Dict[Hashable, float], Dict[str, Any]]:
    """Identify propagation sources from a binary network snapshot.

    Nodes whose ``status_attr`` equals 1 receive initial label +1; all other
    nodes receive -1.  When ``degree_weighted`` is true, failed node ``i``
    instead receives ``max(d_i, 1) ** degree_power`` (optionally normalized),
    which implements LPSI_deg. Labels are updated with

        G(t+1) = alpha * S * G(t) + (1 - alpha) * Y,

    where ``S = D^(-1/2) W D^(-1/2)``.

    ``source_count=None`` implements the original multiple-source rule and
    returns strict local maxima among initially failed nodes. A positive
    ``source_count`` returns that many highest-scoring failed nodes, which is
    used by the repository's known-single-source experiments.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if mode not in {"iterative", "convergent"}:
        raise ValueError("mode must be 'iterative' or 'convergent'")
    if graph_mode not in {"undirected", "directed"}:
        raise ValueError("graph_mode must be 'undirected' or 'directed'")
    if source_count is not None and source_count < 1:
        raise ValueError("source_count must be positive or None")
    if degree_power < 0:
        raise ValueError("degree_power must be non-negative")
    if degree_normalization not in {"none", "max", "mean"}:
        raise ValueError("degree_normalization must be 'none', 'max', or 'mean'")

    nodes = sorted(graph.nodes(), key=str)
    if not nodes:
        return [], {}, {
            "iterations": 0,
            "converged": True,
            "delta": 0.0,
            "degree_weighted": degree_weighted,
        }

    propagation_graph = graph.to_undirected(as_view=False) if graph_mode == "undirected" else graph.copy()
    weight = nx.to_scipy_sparse_array(
        propagation_graph,
        nodelist=nodes,
        dtype=float,
        weight=None,
        format="csr",
    )
    # LPSI uses the edge adjacency matrix without adding self-loops. Treat
    # parallel/reciprocal edges as a single unweighted connection.
    weight = weight.tolil()
    weight.setdiag(0.0)
    weight = weight.tocsr()
    weight.eliminate_zeros()
    if weight.nnz:
        weight.data[:] = 1.0

    degree = np.asarray(weight.sum(axis=1)).ravel()
    inv_sqrt_degree = np.zeros_like(degree, dtype=float)
    nonzero = degree > 0
    inv_sqrt_degree[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    degree_scale = sparse.diags(inv_sqrt_degree, format="csr")
    normalized = degree_scale @ weight @ degree_scale

    failed_mask = np.array(
        [graph.nodes[node].get(status_attr) == 1 for node in nodes], dtype=bool
    )
    initial = np.full(len(nodes), -1.0, dtype=float)
    if degree_weighted:
        # A failed isolated node still needs a positive label.  Using an
        # effective degree of one is also identical to ordinary LPSI on a
        # degree-one node.
        degree_labels = np.maximum(degree, 1.0) ** degree_power
        failed_degree_labels = degree_labels[failed_mask]
        if failed_degree_labels.size and degree_normalization != "none":
            divisor = (
                float(np.max(failed_degree_labels))
                if degree_normalization == "max"
                else float(np.mean(failed_degree_labels))
            )
            if divisor > 0:
                degree_labels = degree_labels / divisor
        initial[failed_mask] = degree_labels[failed_mask]
    else:
        initial[failed_mask] = 1.0
    labels = initial.copy()
    converged = False
    delta = float("inf")

    for iteration in range(1, max_iter + 1):
        updated = alpha * normalized.dot(labels) + (1.0 - alpha) * initial
        delta = float(np.max(np.abs(updated - labels)))
        labels = np.asarray(updated).ravel()
        if mode == "convergent" and delta <= tol:
            converged = True
            break
    else:
        iteration = max_iter

    if mode == "iterative":
        converged = delta <= tol

    scores = {node: float(labels[index]) for index, node in enumerate(nodes)}
    failed = [node for node in nodes if graph.nodes[node].get(status_attr) == 1]
    if not failed:
        return [], scores, {
            "iterations": iteration,
            "converged": converged,
            "delta": delta,
            "degree_weighted": degree_weighted,
        }

    ranked_failed = sorted(failed, key=lambda node: (-scores[node], str(node)))
    if source_count is not None:
        sources = ranked_failed[: min(source_count, len(ranked_failed))]
    else:
        comparison_graph = propagation_graph
        sources = [
            node
            for node in failed
            if all(scores[node] > scores[neighbor] for neighbor in comparison_graph.neighbors(node))
        ]
        # Strict local maxima can be empty on a perfectly tied plateau. Keep a
        # deterministic, non-empty estimate so downstream F1 evaluation works.
        if not sources:
            sources = ranked_failed[:1]
        else:
            sources.sort(key=lambda node: (-scores[node], str(node)))

    stats = {
        "iterations": iteration,
        "converged": converged,
        "delta": delta,
        "degree_weighted": degree_weighted,
    }
    return sources, scores, stats


def lpsi_deg_source_identification(
    graph: nx.Graph,
    alpha: float = 0.5,
    max_iter: int = 5,
    tol: float = 1e-8,
    mode: str = "iterative",
    source_count: Optional[int] = None,
    status_attr: str = "status",
    graph_mode: str = "undirected",
    degree_power: float = 1.0,
    degree_normalization: str = "none",
) -> Tuple[List[Hashable], Dict[Hashable, float], Dict[str, Any]]:
    """Identify sources with the degree-aware LPSI_deg initial labels.

    By default, a failed node ``i`` receives the initial label ``d_i`` while
    a non-failed node receives ``-1``.  ``degree_normalization`` can be set to
    ``"max"`` or ``"mean"`` for scale-sensitive datasets; ``"none"`` keeps
    the direct degree assignment used by the paper-level method definition.
    """
    return lpsi_source_identification(
        graph,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        mode=mode,
        source_count=source_count,
        status_attr=status_attr,
        graph_mode=graph_mode,
        degree_weighted=True,
        degree_power=degree_power,
        degree_normalization=degree_normalization,
    )
