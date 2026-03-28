import py4cytoscape as p4c

# Connect to Cytoscape
p4c.cytoscape_ping()

# Load a .gml file into Cytoscape
def load_gml_network(file_path):
    # Load network from the specified .gml file
    network_suid = p4c.import_network_from_file(file_path, file_type='gml')
    # Apply a layout (optional, but useful for visualization)
    p4c.layout_network(layout_name='circular')
    # Fit the view to the network
    p4c.fit_content()
    return network_suid

# Path to your .gml file
file_path = '/data/william/mlex/visualization/outputs/all_high_5_2/network_graph_phi.gml'
network_suid = load_gml_network(file_path)

print("Network loaded and visualized in Cytoscape with SUID:", network_suid)

print()