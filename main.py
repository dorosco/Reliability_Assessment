import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

def tic():
    global start_time
    start_time = time.perf_counter()

def toc():
    if 'start_time' in globals():
        elapsed_time = time.perf_counter() - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
    else:
        print("Error: 'tic' was not called before 'toc'")
    return elapsed_time

class DistributionTopologyAnalyzer:
    def __init__(self, network_graph, root_node=1):
        """
        Initialize the topology analyzer with a radial network graph
        and ONLY one root node (substation). 
        This class assumes the network is radial and uses NetworkX for graph operations.
        It precomputes paths from the root node to all other nodes for efficient analysis.
        The network graph should be a NetworkX graph with edges representing the distribution lines.
        The root node is typically the substation or main distribution point.
        
        Args:
            network_graph: NetworkX graph representing the radial distribution network
            root_node: The root/substation node (default: 1)
        """
        self.G = network_graph
        self.root = root_node
        self.nodes = list(network_graph.nodes())
        self.edges = list(network_graph.edges())
        
        # Precompute paths from root to all nodes
        self.paths = {node: nx.shortest_path(self.G, self.root, node) for node in self.nodes if node != self.root}
        
        # List of substation nodes (for unit load analysis)
        # this part is not used in the current implementation
        #self.substation_nodes = [self.root] # Root node is the AT/MT 

        # List of nodes wich have as root each substation node
        # this part is not used in the current implementation
        #self.substation_paths = {node: nx.shortest_path(self.G, node, self.root) for node in self.substation_nodes}

    def estimate_interruption_rates_by_node(self):
        """
        Estimate expected number of interruptions per year for each node
        based on the lambda values of the edges in the network.
        
        Returns:
            Dictionary with expected interruptions per year for each node
        """
        N_RP = {}
        D_RP = {}
        D_SW_partial = {}
        for node, path in self.paths.items():
            # Sum the lambda values along the path to the node
            N_RP[node] = sum(self.G[u][v]['lambda'] for u, v in zip(path[:-1], path[1:]))
            D_RP[node] = sum(self.G[u][v]['lambda'] * self.G[u][v]['Tao_RP'] for u, v in zip(path[:-1], path[1:]))
            D_SW_partial[node] = sum(self.G[u][v]['lambda'] * self.G[u][v]['Tao_SW'] for u, v in zip(path[:-1], path[1:]))
        return N_RP, D_RP, D_SW_partial

    def getting_list_of_circuit_breakers_by_node(self):
        """
        Identify which circuit breaker each node belongs to based on the paths.
        
        Returns:
            Dictionary mapping each node to its corresponding circuit breaker
        """
        list_of_circuit_breakers = self.G.neighbors(self.root)
        list_of_circuit_breakers_by_node = {}
        
        for node, path in self.paths.items():
            list_of_circuit_breakers = self.G.neighbors(self.root)
            for cb in list_of_circuit_breakers:
                if cb in path:
                    list_of_circuit_breakers_by_node[node] = cb
                    break
        
        return list_of_circuit_breakers_by_node
    
    def get_edges_by_circuit_breaker(self):
        """
        Integrate the edges belonging to each circuit breaker based on the paths.
        
        Returns:
            Dictionary mapping each circuit breaker to its corresponding edges
        """
        list_of_circuit_breakers = self.G.neighbors(self.root)
        list_edges_by_circuit_breaker = {cb: [] for cb in list_of_circuit_breakers}
        
        for node, path in self.paths.items():
            id_cb = self.getting_list_of_circuit_breakers_by_node().get(node, None)
            if id_cb is not None:
                # Add edges to the corresponding circuit breaker
                edges = [(u, v) for u, v in zip(path[:-1], path[1:])]
                list_edges_by_circuit_breaker[id_cb].extend(edges)
        
        return list_edges_by_circuit_breaker
    
    def get_exclusive_edges_by_circuit_breaker(self):
        """
        Extract the exclusive edges belonging to each circuit breaker using the paths.
        Returns:
            Dictionary mapping each circuit breaker to its exclusive edges
        """
        list_of_circuit_breakers = self.G.neighbors(self.root)
        list_edges_by_circuit_breaker = self.get_edges_by_circuit_breaker()
        
        list_exclusive_edges_by_cb = {cb: [] for cb in list_of_circuit_breakers}
        
        for cb, edges in list_edges_by_circuit_breaker.items():
            # Convert edges to a set for uniqueness
            unique_edges = []
            [
                unique_edges.append(edge) for edge in edges if edge not in unique_edges
            ]
            list_exclusive_edges_by_cb[cb] = unique_edges
        
        return list_exclusive_edges_by_cb
    
    def get_expected_interruption_rates_by_circuit_breaker(self):
        """
        Calculate expected number of interruptions per year for each circuit breaker.
        
        Returns:
            Dictionary mapping each circuit breaker to its expected interruptions per year
        """
        list_exclusive_edges_by_cb = self.get_exclusive_edges_by_circuit_breaker()
        N_CB = {}
        D_CB = {}
        
        for cb, edges in list_exclusive_edges_by_cb.items():
            N_CB[cb] = 0
            D_CB[cb] = 0
            # Iterate through each edge and sum the lambda values for edges that include circuit breakers
            for u, v in edges:
                if self.G.has_edge(u, v):
                    N_CB[cb] += self.G[u][v]['lambda']
                    D_CB[cb] += self.G[u][v]['lambda'] * self.G[u][v]['Tao_SW']
        
        return N_CB, D_CB
    
    def get_expected_interruption_rates_by_node_sw(self):
        """
        Calculate expected number of interruptions per year for each node caused by switching operations.
        
        Returns:
            Dictionary mapping each node to its expected interruptions per year caused by switching operations
        """
        N_RP, D_RP, D_SW_partial = self.estimate_interruption_rates_by_node()
        N_CB, D_CB = self.get_expected_interruption_rates_by_circuit_breaker()
        list_of_circuit_breakers_by_node = self.getting_list_of_circuit_breakers_by_node()
        
        N_SW = {}
        D_SW = {}
        for node, path in self.paths.items():
            # First identify the circuit breaker for the node
            id_cb = list_of_circuit_breakers_by_node.get(node, None)
            if id_cb is not None:
                N_SW[node] = N_CB[id_cb] - N_RP[node]
                D_SW[node] = D_CB[id_cb] - D_SW_partial[node]
        
        return N_SW, D_SW


# Example usage
if __name__ == "__main__":
    # Open the control of time of execution using 'tic' 'toc' functions
    tic()

    # Create a sample radial distribution network (37-node system)
    G = nx.Graph()

    # Getting data from a CSV file to a DataFrame
    # Assuming the CSV file has columns: 'u', 'v', 'lambda', 'Tao_RP', 'Tao_SW'
    #data = pd.read_csv('37_nodes_system_data.csv')
    #data = pd.read_csv('110_nodes_system_data.csv')
    data = pd.read_csv('Alim_508_SEAL_system_data.csv')
    
    # Create edges from the DataFrame
    edges = [
        (
            data['u'].iloc[i],
            data['v'].iloc[i],
            {
                'lambda': data['lambda'].iloc[i],
                'Tao_RP': data['Tao_RP'].iloc[i],
                'Tao_SW': data['Tao_SW'].iloc[i]
            }
        ) for i in range(len(data))
    ]
    G.add_edges_from(edges)
    
    # Getting load data from a CSV file to a DataFrame
    #load_data = pd.read_csv('37_nodes_load_data.csv')
    #load_data = pd.read_csv('110_nodes_load_data.csv')
    load_data = pd.read_csv('Alim_508_SEAL_load_data.csv')

    # Assuming the load data has columns: 'Node', 'NC', 'L_MW'

    for index, row in load_data.iterrows():
        node = row['Node']
        nc = row['NC']
        load_mw = row['L_MW']
        G.nodes[node]['load'] = {'NC': nc, 'L_MW': load_mw}


    print("Created radial distribution network with edges:")
    for u, v, data in G.edges(data=True):
        print(f"Edge {u} -> {v} with lambda {data['lambda']}, Tao_RP {data['Tao_RP']} and Tao_SW {data['Tao_SW']}")

    for node in G.nodes():
        print(f"Node {node} connected to: {list(G.neighbors(node))}")

    # Create paths from root to all nodes
    # Assuming node 1 is the root (substation)
    root = 1  # Substation node

    # Using the DistributionTopologyAnalyzer class
    analyzer = DistributionTopologyAnalyzer(G, root_node=root)
    
    # Estimate interruptions per year for each node
    print("Estimating expected number of interruptions per year for each node using the analyzer:")
    N_RP_analyzer, D_RP_analyzer, D_SW_partial_analyzer = analyzer.estimate_interruption_rates_by_node()
    for node, n_rp in N_RP_analyzer.items():
        print(f"Node {node}: Expected interruptions per year: {n_rp:.2f}")
    for node, d_rp in D_RP_analyzer.items():
        print(f"Node {node}: Expected duration of interruptions per year: {d_rp:.2f}")
    for node, d_sw_partial in D_SW_partial_analyzer.items():
        print(f"Node {node}: Expected duration of switching (partial) interruptions per year: {d_rp:.2f}")
    
    # Getting list of circuit breakers by node
    print("Getting list of circuit breakers by node using the analyzer:")
    list_of_circuit_breakers_by_node_analyzer = analyzer.getting_list_of_circuit_breakers_by_node()
    for node, cb in list_of_circuit_breakers_by_node_analyzer.items():
        print(f"Node {node} belongs to Circuit Breaker {cb}")
    
    # Get edges by circuit breaker
    print("Getting edges by circuit breaker using the analyzer:")
    edges_by_cb = analyzer.get_edges_by_circuit_breaker()
    for cb, edges in edges_by_cb.items():
        print(f"Circuit Breaker {cb}: Edges: {edges}")
    
    # Get exclusive edges by circuit breaker
    print("Getting exclusive edges by circuit breaker using the analyzer:")
    exclusive_edges_by_cb = analyzer.get_exclusive_edges_by_circuit_breaker()
    for cb, edges in exclusive_edges_by_cb.items():
        print(f"Circuit Breaker {cb}: Exclusive Edges: {edges}")
    
    # Get expected interruptions per year by circuit breaker
    print("Getting expected interruptions per year by circuit breaker using the analyzer:")
    expected_interruptions_by_cb, expected_duration_by_cb  = analyzer.get_expected_interruption_rates_by_circuit_breaker()
    for cb, n_cb in expected_interruptions_by_cb.items():
        print(f"Circuit Breaker {cb}: Expected interruptions per year: {n_cb:.2f}")
    for cb, d_cb in expected_duration_by_cb.items():
        print(f"Circuit Breaker {cb}: Expected duration of interruptions per year: {d_cb:.3f}")


    # Get expected interruptions per year by node caused by switching operations
    print("Getting expected interruptions per year by node caused by switching operations using the analyzer:")
    expected_interruptions_by_node_sw, expected_duration_by_node_sw  = analyzer.get_expected_interruption_rates_by_node_sw()
    for node, n_sw in expected_interruptions_by_node_sw.items():
        print(f"Node {node}: Expected interruptions per year caused by switching operations: {n_sw:.2f}")
    for node, d_sw in expected_duration_by_node_sw.items():
        print(f"Node {node}: Expected duration interruptions per year caused by switching operations: {d_sw:.3f}")
    

    # Create a dataframe to store results
    results_df = pd.DataFrame({
        'Node': list(expected_interruptions_by_node_sw.keys()),
        'N_RP (per year)': list(N_RP_analyzer.values()),
        'N_SW (per year)': list(expected_interruptions_by_node_sw.values()),
        'D_RP (per year)': list(D_RP_analyzer.values()),
        'D_SW (per year)': list(expected_duration_by_node_sw.values())
    })
    print("\nResults DataFrame:")
    print(results_df)

    # Savig the results to a CSV file
    output_file = 'radial_distribution_network_results.csv'
    results_df.to_csv(output_file, index=False)

    # Estimate the SAIDI and SAIFI for the network
    numerator_saifi = 0.0
    denominator_saifi = 0.0
    for node, path in analyzer.paths.items():
        nc_node = load_data[load_data['Node']==node]['NC'].iloc[0]
        numerator_saifi += nc_node * (N_RP_analyzer[node] + expected_interruptions_by_node_sw[node]) 
        denominator_saifi += nc_node
    SAIFI = numerator_saifi / denominator_saifi
    
    numerator_saidi = 0.0
    denominator_saidi = 0.0
    for node, path in analyzer.paths.items():
        nc_node = load_data[load_data['Node']==node]['NC'].iloc[0]
        numerator_saidi += nc_node * (D_RP_analyzer[node] + expected_duration_by_node_sw[node]) 
        denominator_saidi += nc_node
    SAIDI = numerator_saidi / denominator_saidi
    print(f"\nEstimated SAIDI: {SAIDI:.2f} hours per year")
    print(f"Estimated SAIFI: {SAIFI:.2f} interruptions per year")

    # Opening outfile in append mode ('a') to print SAIDI and SAIFI
    file_instance = open(output_file, 'a')
    print("\n=============", file = file_instance)
    print(f"Estimated SAIDI: {SAIDI:.2f} hours per year", file = file_instance)
    print(f"Estimated SAIFI: {SAIFI:.2f} interruptions per year", file = file_instance)
    file_instance.close()

    # Close the control of running time
    running_time = toc()

    # Visualize the radial distribution network
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    
    plt.title("Radial Distribution Network Topology")
    plt.show()
