import os
import random
import networkx as nx
from tqdm import tqdm
from create_rr_phi_network import FileManager

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


if __name__ == '__main__':
    graph_file = 'network_graph_phi.gml'
    attribute = 'involvement'
    random_i_d = False
    how_many_randoms = 30
    min_subjects = 5
    min_occurrences = 2
    net_types = ['high']
    years = ['all'] + [str(year) for year in range(2019,2023)]
    folders = [os.path.join('outputs', f'{year}_{net_type}_{min_subjects}_{min_occurrences}') for year in years for
               net_type in net_types]

    for folder in tqdm(folders, desc='Creating random networks'):
        random_out_folder = os.path.join(folder, 'random_networks')
        FileManager.create_dir(random_out_folder)
        G = nx.read_gml(os.path.join(folder, graph_file))
        attr_conta = [data['conta'] for node, data in G.nodes(data=True)]
        attr_values_not_shuffled = [data[attribute] for node, data in G.nodes(data=True)]
        nodes_sorted_by_degree = sorted(G.nodes, key=lambda x: G.degree(x), reverse=True)
        attr_values = [G.nodes[node][attribute] for node in nodes_sorted_by_degree]
        attr_values.sort()

        for i in range(1, how_many_randoms + 1):

            random.shuffle(attr_values)
            for j, node in enumerate(nodes_sorted_by_degree):
                G.nodes[node][attribute] = attr_values[j]
            FileManager.write_gml(G, os.path.join(random_out_folder, f'random{i}_{graph_file}'))

            n = G.number_of_nodes()
            m = G.number_of_edges()
            p = (2 * m) / (n * (n - 1))
            G_er = nx.erdos_renyi_graph(n, p) # Create ER graph with the same number of nodes and an edge probability `p`

            er_nodes = G_er.nodes()
            for j, node in enumerate(er_nodes):
                G_er.nodes[node][attribute] = attr_values_not_shuffled[j]
                G_er.nodes[node]['conta'] = attr_conta[j]
            FileManager.write_gml(G_er, os.path.join(random_out_folder, f'random{i}_ER_{graph_file}'))

            # degree_sequence = [G.degree(node) for node in G.nodes()]
            # G_cm = nx.configuration_model(degree_sequence)
            # G_cm = nx.Graph(G_cm)
            # config_model_nodes_sorted_by_degree = sorted(G_cm.nodes, key=lambda x: G_cm.degree(x), reverse=True)
            # for j, node in enumerate(config_model_nodes_sorted_by_degree):
            #     G_cm.nodes[node][attribute] = attr_values[j]
            # FileManager.write_gml(G_cm, os.path.join(random_out_folder, f'random{i}_CM_{graph_file}'))
