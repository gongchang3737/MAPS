import unittest

import networkx as nx
import numpy as np

from lpsi import lpsi_deg_source_identification, lpsi_source_identification


def set_statuses(graph, failed):
    failed = set(failed)
    nx.set_node_attributes(graph, {node: int(node in failed) for node in graph}, "status")


class LPSITest(unittest.TestCase):
    def test_empty_graph(self):
        sources, scores, stats = lpsi_source_identification(nx.Graph())
        self.assertEqual(sources, [])
        self.assertEqual(scores, {})
        self.assertTrue(stats["converged"])

    def test_single_source_returns_one_failed_node(self):
        graph = nx.star_graph(5)
        set_statuses(graph, graph.nodes())
        sources, _, _ = lpsi_source_identification(graph, source_count=1)
        self.assertEqual(sources, [0])

    def test_multiple_components_produce_local_peaks(self):
        graph = nx.disjoint_union(nx.star_graph(3), nx.star_graph(4))
        set_statuses(graph, graph.nodes())
        sources, _, _ = lpsi_source_identification(graph, source_count=None)
        self.assertEqual(set(sources), {0, 4})

    def test_directed_dag_is_undirected_for_standard_lpsi(self):
        forward = nx.DiGraph([(0, 1), (1, 2), (1, 3)])
        reverse = forward.reverse(copy=True)
        set_statuses(forward, forward.nodes())
        set_statuses(reverse, reverse.nodes())
        _, forward_scores, _ = lpsi_source_identification(forward)
        _, reverse_scores, _ = lpsi_source_identification(reverse)
        self.assertEqual(forward_scores, reverse_scores)

    def test_iterative_solution_converges(self):
        graph = nx.path_graph(5)
        set_statuses(graph, {1, 2, 3})
        _, _, stats = lpsi_source_identification(
            graph, mode="convergent", max_iter=1000, tol=1e-10
        )
        self.assertTrue(stats["converged"])
        self.assertLessEqual(stats["delta"], 1e-10)

    def test_invalid_parameters(self):
        graph = nx.path_graph(2)
        set_statuses(graph, {0})
        with self.assertRaises(ValueError):
            lpsi_source_identification(graph, alpha=1.0)
        with self.assertRaises(ValueError):
            lpsi_source_identification(graph, max_iter=0)


class LPSIDegTest(unittest.TestCase):
    def test_degree_labels_change_failed_hub_score(self):
        graph = nx.star_graph(3)
        set_statuses(graph, {0})
        _, lpsi_scores, _ = lpsi_source_identification(
            graph, source_count=1, max_iter=1
        )
        sources, degree_scores, stats = lpsi_deg_source_identification(
            graph, source_count=1, max_iter=1
        )
        self.assertEqual(sources, [0])
        self.assertGreater(degree_scores[0], lpsi_scores[0])
        self.assertTrue(stats["degree_weighted"])

    def test_max_normalization_matches_lpsi_with_one_failed_node(self):
        graph = nx.star_graph(4)
        set_statuses(graph, {0})
        _, lpsi_scores, _ = lpsi_source_identification(graph, source_count=1)
        _, degree_scores, _ = lpsi_deg_source_identification(
            graph, source_count=1, degree_normalization="max"
        )
        for node in graph:
            self.assertAlmostEqual(degree_scores[node], lpsi_scores[node])

    def test_multiple_components_return_degree_aware_local_peaks(self):
        graph = nx.disjoint_union(nx.star_graph(3), nx.star_graph(4))
        set_statuses(graph, graph.nodes())
        sources, _, _ = lpsi_deg_source_identification(graph, source_count=None)
        self.assertEqual(set(sources), {0, 4})

    def test_invalid_degree_parameters(self):
        graph = nx.path_graph(2)
        set_statuses(graph, {0})
        with self.assertRaises(ValueError):
            lpsi_deg_source_identification(graph, degree_power=-1)
        with self.assertRaises(ValueError):
            lpsi_deg_source_identification(graph, degree_normalization="bad")


if __name__ == "__main__":
    unittest.main()
