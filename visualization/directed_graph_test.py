import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from create_rr_phi_network import DataPreparer


df = DataPreparer.read_data(r"/data/pcpe")

df = df.loc[df['ANO_LANCAMENTO']=='2019',]

G = nx.DiGraph()
od_accounts = set(df["CONTA_OD"].dropna().astype(str))
cpf_cnpj_titular_accounts = set(df["CPF_CNPJ_TITULAR"].dropna().astype(str))

for _, row in df.iterrows():
    conta_titular = str(row["CONTA_TITULAR"])
    conta_od = str(row["CONTA_OD"])
    natureza = row["NATUREZA_LANCAMENTO"]
    cpf_cnpj_titular = str(row["CPF_CNPJ_TITULAR"])

    edge_attrs = {
        "weight": row["VALOR_TRANSACAO"],
        "I_d": row["I-d"],
        "I_e": row["I-e"],
        "IV_n": row["IV-n"],
        "DATA_LANCAMENTO": row["DATA_LANCAMENTO"],
        "NATUREZA_LANCAMENTO": natureza
    }

    if conta_od == "EMPTY":
        if natureza == "C":
            G.add_edge(conta_titular, conta_titular, **edge_attrs)
        elif natureza == "D":
            G.add_edge(conta_titular, cpf_cnpj_titular, **edge_attrs)
    else:
        if natureza == "C":
            G.add_edge(conta_od, conta_titular, **edge_attrs)
        elif natureza == "D":
            G.add_edge(conta_titular, conta_od, **edge_attrs)

for node in G.nodes():
    G.nodes[node]["name"] = node
    G.nodes[node]["IS_OD"] = node in od_accounts
    G.nodes[node]["IS_CPF_CNPJ_TITULAR"] = node in cpf_cnpj_titular_accounts

output_gml = "directed_graph_test_2019.gml"
nx.write_gml(G, output_gml)
print(f"Graph saved as {output_gml}")

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=10, node_color="lightblue", edge_color="gray", arrows=True)

edge_labels = {(u, v): f"{d['weight']} ({d['DATA_LANCAMENTO']})" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Transaction Network with Attributes")
plt.show()
