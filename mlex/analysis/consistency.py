from typing import List
from mlex import TransactionFlow, TransactionMultiDigraph, Trail
from mlex.utils.datastructures import DoublyLinkedList
import networkx as nx


class IntraTimestamp:
    @classmethod
    def balance_consistency(cls, T: List[List[TransactionFlow]]):
        """
        Identifies intra timestamps consistencies and inconsistnecies of the provided transaction flows `T`.

        Constructs a `TransactionMultiDigraph` for every timestamp and, if possible, creates an open/closed Eulerian trail. Each Eulerian trail is then appended to a a `List` `S`.
        When it is not possible to create an Eulerian trail, the `TransactionMultiDigraph` is then passed to `__inconsistent_components` in order to get the asymetric components `A` and Eulerian components `D`  

        Parameters
        ----------
        - T: `List[List[TransactionFlow]]` A nested list of `TransactionFlow` objects representing separate timestamps of bank transactions.

        Returns
        -------
        - `S` : `List[List[int]]` nested list of integers representing the ordering of indices of the consistent timestamps.
        - `W` : `List[(A, D)]` list of pairs of indices of inconsistent timestamps
                - `A` is a set of tuples `(A_1, A_2)` for the components of a inconsistent timestamp
                    - Set `A_1` contains the indices of transaction flows that belong to a balance node `u` with
                    `abs(in_degree(u) - out_degree(u)) > 1`
                    - Set `A_2` contains the indices of transaction flows that belong to a balance node `v` with
                    `abs(in_degree(v) - out_degree(v)) <= 1`
                - `D` is a set of tuples `set(Tuple(int))` of the indices ordering of Eulerian components of a transaction flows, within a inconsistent timestamp
        """

        if not T:
            raise ValueError("At least one TransactionFlow list must be provided.")
        S, W = [], []
        C: DoublyLinkedList = None
        for i, T_i in enumerate(T):
            if len(set([t.timestamp for t in T_i])) != 1:
                raise ValueError(
                    f"Transaction sequence at index {i} has multiple timestamps. Each sequence must have unique timestamp."
                )
            G_i = TransactionMultiDigraph.create_multidigraph(transaction_flows=T_i)
            if nx.is_weakly_connected(G_i):
                if G_i.is_pseudo_symmetric():
                    s = next(iter(G_i))
                    C = Trail.eulerian_circuit(G=G_i, s=s)
                    S.append(list(C))
                else:
                    O = G_i.get_asymmetric_difference(k=1)
                    P = G_i.get_pseudosymmetric_nodes()
                    # Checks whether there is a single pair of asymmetric nodes with difference of in degree and out degree equal to 1,
                    # and the total number of pseudosymetric nodes is the total number of nodes - 2.
                    if len(O) == 1 and len(P) == len(G_i) - 2:
                        # There is an open eulerian trail in the multidigraph G_i
                        q = 0
                        ((u, v),) = O
                        G_i.add_edge(v, u, key=q)
                        C = Trail.eulerian_circuit(G=G_i, s=v, e=(v, u, q))
                        while C.head.value != q:
                            C.right_shift()
                        C.remove(key=q, key_selector=lambda node: node.value)
                        S.append(list(C))
                    else:
                        W_i = cls.__inconsistent_components(G_i)
                        W.append(W_i)
            else:
                W_i = cls.__inconsistent_components(G_i)
                W.append(W_i)
        return S, W

    @classmethod
    def __inconsistent_components(cls, G):
        A, D = set(), set()
        for Z_k in [
            G.subgraph(Z_k).copy() for Z_k in nx.weakly_connected_components(G)
        ]:
            if all([Z_k.in_degree[node] == Z_k.out_degree[node] for node in Z_k]):
                s = next(iter(Z_k))
                C = Trail.eulerian_circuit(G=Z_k, s=s)
                D.add(tuple(C))
            else:
                O = {
                    (u, v)
                    for u in Z_k
                    if Z_k.in_degree[u] + 1 == Z_k.out_degree[u]
                    for v in Z_k
                    if Z_k.in_degree[v] == Z_k.out_degree[v] + 1
                }
                P = {u for u in Z_k if Z_k.in_degree[u] == Z_k.out_degree[u]}
                if len(O) == 1 and len(P) == len(Z_k) - 2:
                    q = 0
                    ((u, v),) = O
                    Z_k.add_edge(v, u, key=q)
                    C = Trail.eulerian_circuit(G=Z_k, s=v, e=(v, u, q))
                    while C.head.value != q:
                        C.right_shift()
                    C.remove(key=q, key_selector=lambda node: node.value)
                    D.add(tuple(C))
                else:
                    A_1 = frozenset(
                        [
                            key
                            for node in Z_k
                            if abs(Z_k.in_degree[node] - Z_k.out_degree[node]) > 1
                            for _, _, key in (
                                list(Z_k.in_edges(node, keys=True))
                                + list(Z_k.out_edges(node, keys=True))
                            )
                        ]
                    )
                    A_2 = frozenset(
                        [
                            key
                            for node in Z_k
                            if abs(Z_k.in_degree[node] - Z_k.out_degree[node]) <= 1
                            for _, _, key in (
                                list(Z_k.in_edges(node, keys=True))
                                + list(Z_k.out_edges(node, keys=True))
                            )
                        ]
                    )

                    A.add((A_1, A_2.difference(A_1)))
        W = (A, D)
        return W
