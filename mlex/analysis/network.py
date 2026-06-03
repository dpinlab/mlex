import networkx as nx
from mlex.utils.schema import TransactionFlow
from typing import List
from mlex.utils.datastructures import DoublyLinkedList
from collections import deque, defaultdict


class DisconnectedMultiDigraphError(Exception):
    pass


class TransactionMultiDigraph:
    @classmethod
    def create_multidigraph(
        cls,
        transaction_flows: List[TransactionFlow],
    ) -> nx.MultiDiGraph:
        if len(transaction_flows) == 0:
            raise ValueError(
                "No transaction flow received. Transaction flow network must have at least one flow."
            )
        G = nx.MultiDiGraph()
        for flow in transaction_flows:
            G.add_edge(
                u_for_edge=flow.balance_amount - flow.tx_value,
                v_for_edge=flow.balance_amount,
                key=flow.tx_index,
                **{
                    "tx_value": flow.tx_value,
                    "timestamp": flow.timestamp,
                },
            )
        return G


class Trail:
    # The following implementation is an adaptation of the Hierholzer Algotithm
    # present on the book Graphs, Networks and Algorithms (Jugnickel), Section 2.3
    @classmethod
    def eulerian_circuit(
        cls, G: nx.MultiDiGraph, s: int, e: tuple[int, int, int] = None
    ):
        if not nx.is_strongly_connected(G):
            raise DisconnectedMultiDigraphError(
                "Cannot create Eulerian Circuit on a not strongly connected multidigraph."
            )
        # The following data strucutures are an adaptation of those described in
        # Graphs, Networks and Algorithms (Jugnickel) Section 2.3.1

        # Final eulerian trail and Node stack containing the nodes currently present in the trail
        K, L = DoublyLinkedList[dict](), deque()
        U = set()
        # A mapping from vertices v to edge indices q already within the current circuit K
        f = defaultdict(type(None))
        # Preserves the structure of the original against the edge deletions
        G_c = G.copy()
        if e:
            s, v, q = e
            G_c.remove_edge(u=s, v=v, key=q)
            K.append(q)
            U = U.union({s})
            L.append(s)
            f[s] = K.tail
            s = v
        U = U.union({s})
        L.append(s)
        cls._trace(G=G_c, s=s, U=U, f=f, L=L, C=K)
        while L:
            u = L.pop()
            C = DoublyLinkedList[dict]()
            cls._trace(G=G_c, s=u, U=U, f=f, L=L, C=C)
            # This operation will splice the obtained closed trail in front of the edge pointed by the node u
            K.splice(node=f[u], dll=C)

        # After adding the temporary edge between the non pseudosymmetric nodes u and v
        # we need to shift the trail to the right until the extra edged appears at the head
        # and remove it in order to start at u and end at v
        # Graphs Theory and its Applications (Gross et al.) Section 6.1
        return K

    # The following implementation constructs a closed trail in a digraph and is adapted from
    # Graphs, Networks and Algorithms (Jugnickel), Section 2.3
    @classmethod
    def _trace(
        cls,
        G: nx.MultiDiGraph,
        s: int,
        U: set,
        f: dict,
        L: deque,
        C: DoublyLinkedList,
    ):
        while G.out_edges(s):
            e = next(iter(G.out_edges(s,keys=True)))
            u,v,k = e
            G.remove_edge(u=u,v=v,key=k)
            C.append(k)
            if not f[s]:
                f[s] = C.tail
            s = v
            if s not in U:
                L.append(s)
                U.add(s)

    # This method checks for pseudosymmetry.
    # If there is at most 2 nodes u and v with outdegree(u) = indegree(u) + 1 and indegree(v) = outdegree(v) + 1,
    # and the other nodes are pseudosymmetric,
    # there is an open eulerian trail between u and v 
    # Graph Theory and its Applications (Gross et al.) Section 6.1
    @classmethod
    def is_open_trail(cls, G: nx.MultiDiGraph):
        s = next(iter(G.nodes()))
        e = None
        degrees = [(n, G.in_degree(n), G.out_degree(n)) for n in G.nodes()]
        if not all([indegree == outdegree for _, outdegree, indegree in degrees]):
            unmatched_outdegrees = [
                indegree + 1 == outdegree for _, indegree, outdegree in degrees
            ]
            unmatched_indegrees = [
                outdegree + 1 == indegree for _, indegree, outdegree in degrees
            ]
            if not (sum(unmatched_outdegrees) == 1 and sum(unmatched_indegrees) == 1):
                raise DisconnectedMultiDigraphError(
                    "Cannot create Eulerian Circuit on a not strongly connected multidigraph."
                )
            u, _, _ = degrees[unmatched_outdegrees.index(True)]
            v, _, _ = degrees[unmatched_indegrees.index(True)]
            G.add_edge(u, v, key=0)
            s = u
            e = (u, v, 0)
        return s, e
