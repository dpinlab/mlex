import abc

import networkx as nx
import numpy as np
from operator import attrgetter

from .schema import Transaction


class NetworkPathStrategy(abc.ABC):
    @abc.abstractmethod
    def create_digraph(self, root_node: dict, values: np.ndarray, backwards=True):
        pass

    @abc.abstractmethod
    def find_path(self, G: nx.DiGraph, source: int):
        pass

    @abc.abstractmethod
    def get_transactions(self, G, path):
        pass

    @abc.abstractmethod
    def get_source(node, backwards=False):
        pass


class HamiltonianPathStrategy(NetworkPathStrategy):
    def create_digraph(
        self, root_node: Transaction, values: np.ndarray, backwards=True
    ):
        get_previous_balance = attrgetter("previous_balance")
        get_balance_amount = attrgetter("balance_amount")
        source_attr, target_attr = (
            (get_previous_balance, get_balance_amount)
            if backwards
            else (get_balance_amount, get_previous_balance)
        )
        D = nx.DiGraph()
        D.add_node(0, transaction=root_node)
        D.add_nodes_from(enumerate(map(lambda x: {"transaction": x}, values), start=1))
        target_func = np.frompyfunc(target_attr, 1, 1)
        target_values = target_func(
            np.array([data["transaction"] for _, data in D.nodes(data=True)])
        ).astype(int)
        for node, data in D.nodes(data=True):
            source_value = source_attr(data["transaction"])
            target_nodes = np.where(target_values == source_value)[0]
            edges = [(node, target_node) for target_node in target_nodes]
            D.add_edges_from(edges)
        return D

    def find_path(self, G, source):
        if len(list(nx.weakly_connected_components(G))) > 1:
            return None

        n = len(G.nodes)
        path = [source]
        visited = {source}

        def backtrack(current):
            if len(path) == n:
                return path

            for neighbor in G.successors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)

                    result = backtrack(neighbor)
                    if result:
                        return result

                    path.pop()
                    visited.remove(neighbor)
            return None

        return backtrack(source)

    def get_transactions(self, G, path):
        return [G.nodes[i]["transaction"] for i in path]

    def get_source(self, node, backwards=False):
        return 0


class EulerianPathStrategy(NetworkPathStrategy):
    def create_digraph(
        self, root_node: Transaction, values: np.ndarray, backwards=True
    ):
        get_previous_balance = attrgetter("previous_balance")
        get_balance_amount = attrgetter("balance_amount")
        source_attr, target_attr = (
            (get_balance_amount, get_previous_balance)
            if backwards
            else (get_previous_balance, get_balance_amount)
        )

        D = nx.MultiDiGraph()
        D.add_node(source_attr(root_node))

        for transaction in values:
            transaction_source = source_attr(transaction)
            D.add_node(transaction_source)

        D.add_edge(
            source_attr(root_node), target_attr(root_node), transaction=root_node
        )

        for transaction in values:
            transaction_source = source_attr(transaction)
            transaction_target = target_attr(transaction)
            D.add_edge(transaction_source, transaction_target, transaction=transaction)
        return D

    def find_path(self, G, source):
        if not nx.has_eulerian_path(G):
            return None

        in_deg = G.in_degree(source)
        out_deg = G.out_degree(source)

        is_circuit = nx.is_eulerian(G)
        is_valid_start = is_circuit or (out_deg == in_deg + 1)

        if not is_valid_start:
            return None
        try:
            path_iterator = nx.eulerian_path(G, source=source, keys=True)
            return [(u, v, k) for u, v, k in path_iterator]
        except nx.NetworkXError:
            return None

    def get_transactions(self, G, path):
        return [G.edges[u, v, k]["transaction"] for u, v, k in path]

    def get_source(self, node: Transaction, backwards=False):
        return node.previous_balance if not backwards else node.balance_amount
