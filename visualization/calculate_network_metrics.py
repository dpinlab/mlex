import os
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def compute_metrics(G, network_name):
    """Compute metrics and return as a dictionary, optimized for performance."""
    metrics = {
        "Network": network_name,
        "Node": [],
        "Degree": [],
        "Betweenness": [],
        "Clustering": []
    }

    degree_dict = dict(G.degree())

    clustering_dict = nx.clustering(G)

    betweenness_dict = nx.betweenness_centrality(G)

    # Populate node metrics once
    for node, data in G.nodes(data=True):
        metrics["Node"].append(data.get('conta', node))
        metrics["Degree"].append(degree_dict[node])
        metrics["Betweenness"].append(betweenness_dict[node])
        metrics["Clustering"].append(clustering_dict[node])

    if nx.is_connected(G):
        avg_shortest_path = nx.average_shortest_path_length(G)
    else:
        largest_component = max(nx.connected_components(G), key=len)
        G_largest = G.subgraph(largest_component).copy()
        avg_shortest_path = nx.average_shortest_path_length(G_largest)

    # Populate mean metrics
    mean_metrics = {
        "Network": network_name,
        "Mean_Degree": np.mean(metrics["Degree"]),
        "Mean_Betweenness": np.mean(metrics["Betweenness"]),
        "Mean_Clustering": np.mean(metrics["Clustering"]),
        "Average_Shortest_Path": avg_shortest_path
    }

    return metrics, mean_metrics


if __name__ == '__main__':
    graph_file = 'less_EGO_network_graph_phi.gml'
    output_csv = 'network_metrics.csv'
    random_net = True
    num_randoms = 30
    min_subjects = 5
    min_occurrences = 2
    net_types = ['high']
    years = ['all'] + [str(year) for year in range(2019,2023)]
    folders = [os.path.join('outputs', f'{year}_{net_type}_{min_subjects}_{min_occurrences}') for year in years for
               net_type in net_types]

    all_node_metrics = []  # List to store per-node metrics for all networks
    all_mean_metrics = []  # List to store mean metrics for each network

    for folder in tqdm(folders, desc='Creating random networks'):
        folder_name = os.path.basename(os.path.normpath(folder))
        G = nx.read_gml(os.path.join(folder, graph_file))

        # Calculate and store metrics for the original graph
        node_metrics, mean_metrics = compute_metrics(G, network_name=f"Real_{folder_name}")
        all_node_metrics.append(pd.DataFrame(node_metrics))  # Save per-node metrics
        all_mean_metrics.append(mean_metrics)  # Save mean metrics

        if random_net:
            random_out_folder = os.path.join(folder, 'random_networks')
            for i in range(1, num_randoms + 1):
                # G_r = nx.read_gml(os.path.join(random_out_folder, f'random{i}_{graph_file}'))
                # node_metrics, mean_metrics = compute_metrics(G_r, network_name=f"Random_{i}_{folder_name}")
                # all_node_metrics.append(pd.DataFrame(node_metrics))  # Save per-node metrics
                # all_mean_metrics.append(mean_metrics)  # Save mean metrics

                G_er = nx.read_gml(os.path.join(random_out_folder, f'random{i}_ER_{graph_file}'))
                node_metrics, mean_metrics = compute_metrics(G_er, network_name=f"ER_{i}_{folder_name}")
                all_node_metrics.append(pd.DataFrame(node_metrics))  # Save per-node metrics
                all_mean_metrics.append(mean_metrics)  # Save mean metrics

    # Combine all metrics and save to CSV
    node_metrics_df = pd.concat(all_node_metrics, ignore_index=True)
    node_metrics_df.to_csv(os.path.join('outputs', f"{output_csv}_detailed.csv"), index=False)
    mean_metrics_df = pd.DataFrame(all_mean_metrics)
    mean_metrics_df.to_csv(os.path.join('outputs', output_csv), index=False)
