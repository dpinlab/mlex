import networkx as nx
from mlex.utils.schema import TransactionFlow
from typing import List
import json
from mlex.utils.datastructures import DoublyLinkedList
from collections import deque, defaultdict


class NoEulerianTrailError(Exception):
    def __init__(self, multidigraph: nx.MultiDiGraph):
        self.multidigraph = multidigraph
        self.message = f"Transaction MultiDigraph is not pseudosymmetric and has no open eulerian trail"
        super().__init__(self.message)


class DisconnectedMultiDigraphError(Exception):
    pass


class TransactionMultiDigraph:
    @classmethod
    def create_multigraph(
        cls,
        transaction_flows: List[TransactionFlow],
    ) -> nx.MultiDiGraph:
        if len(transaction_flows) == 0:
            raise ValueError(
                "No transaction flow received. Transaction flow network must have at least one flow."
            )
        D = nx.MultiDiGraph()
        for flow in transaction_flows:
            D.add_edge(
                u_for_edge= flow.balance_amount - flow.tx_value,
                v_for_edge=flow.balance_amount,
                key=flow.tx_index,
                **{
                    "tx_value": flow.tx_value,
                    "timestamp": flow.timestamp,
                },
            )
        return D


class Trail:
    # The following implementation is an adaptation of the Hierholzer Algotithm
    # present on the book Graphs, Networks and Algorithms (Jugnickel), Section 2.3
    @classmethod
    def eulerian_trail(cls, D: nx.MultiDiGraph):
        if not nx.is_weakly_connected(D):
            raise DisconnectedMultiDigraphError(
                "Cannot create Eulerian Trail on a disconnected multigraph."
            )
        # The following data strucutures are an adaptation of those described in
        # Graphs, Networks and Algorithms (Jugnickel) Section 2.3.1
        
        # Node list containing the nodes currently present in the eulerian trail
        L = deque()
        # Final eulerian trail
        K = DoublyLinkedList[dict]()
        # Preserves the structure of the original against the edge deletions
        D_c = D.copy()
        # A mapping for an edge within the current trail with end in a node u 
        e_p = defaultdict(type(None))
        # Starting node s and a flag determining wether the eulerian trail is open
        s, open = cls._open_trail(D_c)
        # Mappings to keep track of used nodes and new edges for the trail
        used = dict({v: False for v in D_c})
        new = dict({k: True for (_, _, k) in D_c.edges(keys=True)})
        used[s] = True
        L.append(s)
        cls._trace(s, e_p, D_c, new, used, K, L)
        while L:
            u = L.pop()
            C = DoublyLinkedList[dict]()
            cls._trace(u, e_p, D_c, new, used, C, L)
            # This operation will splice the obtained closed trail in front of the edge pointed by the node u 
            K.splice(e_p[u], C)

        # After adding the temporary edge between the non pseudosymmetric nodes u and v
        # we need to shift the trail to the right until the extra edged appears at the head
        # and remove it in order to start at u and end at v
        # Graphs Theory and its Applications (Gross et al.) Section 6.1
        if open:
            while K.head.value["key"] != -1:
                K.right_shift()
            K.remove(-1, lambda node: node.value["key"])
        return K.itemize(lambda node: node.value["key"])

    # The following implementation constructs a closed trail in a digraph and is adapted from
    # Graphs, Networks and Algorithms (Jugnickel), Section 2.3
    @classmethod
    def _trace(
        cls,
        v,
        e_p,
        D: nx.MultiDiGraph,
        new: dict,
        used: dict,
        C: DoublyLinkedList,
        L: deque,
    ):
        item = next(
            (
                dict({"source": v, "target": w, "key": k})
                for (v, w, k) in deque(D.edges(v, keys=True), maxlen=1)
            ),
            {},
        )
        A_v = [item] if item else None
        while A_v:
            e = A_v[0]
            D.remove_edge(*e.values())
            if new[e["key"]]:
                C.append(e)
                if not e_p[v]:
                    e_p[v] = C.search(e["key"], lambda node: node.value["key"])
                new[e["key"]] = False
                v = e["target"]
                if not used[v]:
                    L.append(v)
                    used[v] = True
                item = next(
                    (
                        dict({"source": v, "target": w, "key": k})
                        for (v, w, k) in deque(D.edges(v, keys=True), maxlen=1)
                    ),
                    {},
                )
                A_v = [item] if item else None

    # This method checks for pseudosymmetry.
    # If there is at most 2 non pseudosymmetric nodes u and v, there is an open eulerian trail between u and v if and only if
    # outdegree(u) = indegree(u) + 1 and indegree(v) = outdegree(v) + 1
    # Graph Theory and its Applications (Gross et al.) Section 6.1
    @classmethod
    def _open_trail(cls, D: nx.MultiDiGraph):
        s = next(iter(D.nodes()))
        open = False
        degrees = [(n, D.in_degree(n), D.out_degree(n)) for n in D.nodes()]
        if not all([indegree == outdegree for _, outdegree, indegree in degrees]):
            unmatched_outdegrees = [
                indegree + 1 == outdegree for _, indegree, outdegree in degrees
            ]
            unmatched_indegrees = [
                outdegree + 1 == indegree for _, indegree, outdegree in degrees
            ]
            if not (sum(unmatched_outdegrees) == 1 and sum(unmatched_indegrees) == 1):
                raise NoEulerianTrailError(D)
            open = True
            u, _, _ = degrees[unmatched_outdegrees.index(True)]
            v, _, _ = degrees[unmatched_indegrees.index(True)]
            D.add_edge(u, v, key=-1)
            s = u
            open = True
        return s, open
