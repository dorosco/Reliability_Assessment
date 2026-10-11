import pulp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class DistributionTopologyAnalyzer:
    def __init__(self, network_graph, root_node=1):
        """
        Initialize the topology analyzer with a radial network graph
        
        Args:
            network_graph: NetworkX graph representing the radial distribution network
            root_node: The root/substation node (default: 1)
        """
        self.G = network_graph
        self.root = root_node
        self.nodes = list(network_graph.nodes())
        self.edges = list(network_graph.edges())
        
        # Add reverse edges for modeling purposes (power can flow both ways in model)
        self.directed_edges = self.edges + [(v, u) for (u, v) in self.edges]
        
        # Precompute paths from root to all nodes
        self.paths = {node: nx.shortest_path(self.G, self.root, node) for node in self.nodes if node != self.root}
        
        # List of substation nodes (for unit load analysis)
        self.substation_nodes = [self.root] # Root node is the AT/MT 

        # List of nodes wich have as root each substation node
        # this part is not used in the current implementation
        #self.substation_paths = {node: nx.shortest_path(self.G, node, self.root) for node in self.substation_nodes}

        # List of load nodes (all nodes except root)
        self.load_nodes = [node for node in self.nodes if node != self.root]


    def create_unit_load_model(self, load_node):
        """
        Create LP model for unit load at specified node
        
        Args:
            load_node: Node where 1 pu load is placed (others set to 0)
            
        Returns:
            A PuLP LP problem object
        """
        # Create LP problem
        prob = pulp.LpProblem(f"Topology_Characterization_Node_{load_node}", pulp.LpMinimize)
        
        # Variables
        # Power flow on each edge (integer, can be positive or negative)
        F = pulp.LpVariable.dicts("Flow", self.directed_edges, lowBound=0, upBound=1, cat='Integer')
        
        # Injection at each substation node
        G = pulp.LpVariable.dicts("Injection", self.nodes, lowBound=0, upBound=1, cat='Integer')
        
        # Objective function: minimize total power flow
        for (u,v) in self.edges:
            prob += pulp.lpSum(F[(u, v)] + F[(v, u)])


        # Constraints
        # Power flow variables
        # 
        # Nodal power balance equations


        for node in self.nodes:
            if node == self.root:
                # Root node balances total system load
                prob += pulp.lpSum(F[(u, v)] for (u, v) in self.directed_edges if v == node) - \
                        pulp.lpSum(F[(u, v)] for (u, v) in self.directed_edges if u == node) == \
                        (-1 if node == load_node else 0), f"Power_balance_{node}"
            else:
                # Other nodes balance their own load
                prob += pulp.lpSum(F[(u, v)] for (u, v) in self.directed_edges if v == node) - \
                        pulp.lpSum(F[(u, v)] for (u, v) in self.directed_edges if u == node) == \
                        (1 if node == load_node else 0), f"Power_balance_{node}"
        #         
        # Putting 1 pu load at the specified node
        # P = pulp.LpVariable.dicts("Power", self.directed_edges, lowBound=-1e6, upBound=1e6, cat='Continuous')
        # V = pulp.LpVariable.dicts("Voltage", self.nodes, lowBound=0.9, upBound=1.1, cat='Continuous')
        # # Set voltage at root node to 1.0 pu
        # prob += V[self.root] == 1.0, "Root_voltage"
        
        # # Nodal power balance equations
        # for node in self.nodes:
        #     if node == self.root:
        #         # Root node balances total system load
        #         prob += pulp.lpSum(P[(u, v)] for (u, v) in self.directed_edges if v == node) - \
        #                 pulp.lpSum(P[(u, v)] for (u, v) in self.directed_edges if u == node) == \
        #                 (-1 if node == load_node else 0), f"Power_balance_{node}"
        #     else:
        #         # Other nodes balance their own load
        #         prob += pulp.lpSum(P[(u, v)] for (u, v) in self.directed_edges if v == node) - \
        #                 pulp.lpSum(P[(u, v)] for (u, v) in self.directed_edges if u == node) == \
        #                 (1 if node == load_node else 0), f"Power_balance_{node}"


        # Root node voltage fixed at 1.0 pu
        prob += V[self.root] == 1.0, "Root_voltage"
        
        # Power balance at each node
        for node in self.nodes:
            if node == self.root:
                # Root node balances total system load
                prob += pulp.lpSum(P[(u, v)] for (u, v) in self.directed_edges if v == node) - \
                        pulp.lpSum(P[(u, v)] for (u, v) in self.directed_edges if u == node) == \
                        (-1 if node == load_node else 0), f"Power_balance_{node}"
            else:
                # Other nodes balance their own load
                prob += pulp.lpSum(P[(u, v)] for (u, v) in self.directed_edges if v == node) - \
                        pulp.lpSum(P[(u, v)] for (u, v) in self.directed_edges if u == node) == \
                        (1 if node == load_node else 0), f"Power_balance_{node}"
        
        # Voltage drop across edges (simplified linear approximation)
        for (u, v) in self.edges:
            R = self.G[u][v].get('resistance', 0.01)  # Default resistance if not specified
            X = self.G[u][v].get('reactance', 0.01)   # Default reactance if not specified
            prob += V[u] - V[v] == R * P[(u, v)] + X * P[(v, u)], f"Voltage_drop_{u}_{v}"
            prob += V[v] - V[u] == R * P[(v, u)] + X * P[(u, v)], f"Voltage_drop_{v}_{u}"
        
        # Radiality constraints (ensure power flows from root to load)
        for node in self.nodes:
            if node != self.root:
                path = self.paths[node]
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    prob += P[(u, v)] >= 0, f"Radial_flow_{u}_{v}_to_{node}"
        
        print("Creating model for unit load at node:", load_node)
        print(prob)  # Debug: print the problem formulation
        print(prob.constraints)  # Debug: print constraints
        return prob
    
    def analyze_topology(self):
        """
        Analyze network topology by solving unit load models for all nodes
        
        Returns:
            Dictionary with results for each node's unit load scenario
        """
        results = {}
        
        for load_node in self.nodes:
            if load_node == self.root:
                continue  # Skip root node as load location
            
            print(f"Solving for unit load at node {load_node}...")
            prob = self.create_unit_load_model(load_node)
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            
            if pulp.LpStatus[prob.status] == 'Optimal':
                # Collect power flow results
                flows = {(u, v): pulp.value(P[(u, v)]) for (u, v) in self.directed_edges}
                
                # Determine which edges carry power (for topology characterization)
                active_edges = [(u, v) for (u, v) in self.edges 
                               if abs(flows[(u, v)]) > 1e-3 or abs(flows[(v, u)]) > 1e-3]
                
                results[load_node] = {
                    'status': 'Optimal',
                    'active_edges': active_edges,
                    'power_flows': flows,
                    'voltage_profile': {node: pulp.value(V[node]) for node in self.nodes}
                }
            else:
                results[load_node] = {'status': pulp.LpStatus[prob.status]}
        
        return results

    def visualize_results(self, results, load_node):
        """
        Visualize the power flows for a specific load node scenario
        
        Args:
            results: Results dictionary from analyze_topology()
            load_node: Node to visualize
        """
        if load_node not in results or results[load_node]['status'] != 'Optimal':
            print(f"No valid results for node {load_node}")
            return
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.G)
        
        # Draw the network
        nx.draw_networkx_nodes(self.G, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(self.G, pos, edge_color='gray', width=1, alpha=0.5)
        nx.draw_networkx_labels(self.G, pos)
        
        # Highlight active edges and show power flow direction
        active_edges = results[load_node]['active_edges']
        flows = results[load_node]['power_flows']
        
        for (u, v) in self.edges:
            if (u, v) in active_edges or (v, u) in active_edges:
                flow_val = flows[(u, v)] - flows[(v, u)]
                width = 2 + 2 * abs(flow_val)
                style = 'solid' if flow_val >= 0 else 'dashed'
                nx.draw_networkx_edges(self.G, pos, edgelist=[(u, v)], 
                                      width=width, edge_color='red', 
                                      style=style, arrows=True)
        
        # Highlight the load node
        nx.draw_networkx_nodes(self.G, pos, nodelist=[load_node], 
                              node_color='red', node_size=700)
        
        plt.title(f"Power flows with 1 pu load at node {load_node}")
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create a sample radial distribution network (8-node system)
    G = nx.Graph()
    edges = [
        (1, 2, {'lambda': 0.5, 'Tao_RP': 1.0, 'Tao_SW': 0.15}),  # First main feeder
        (2, 3, {'lambda': 0.2, 'Tao_RP': 2.0, 'Tao_SW': 0.50}),  # Lateral
        (2, 4, {'lambda': 0.1, 'Tao_RP': 4.0, 'Tao_SW': 0.25}),  # Lateral
        (1, 5, {'lambda': 0.3, 'Tao_RP': 1.5, 'Tao_SW': 0.20}),  # Another main branch
        (5, 6, {'lambda': 0.4, 'Tao_RP': 3.0, 'Tao_SW': 0.60}),  # Lateral
    ]
    G.add_edges_from(edges)
    
    print("Created radial distribution network with edges:")
    for u, v, data in G.edges(data=True):
        print(f"Edge {u} -> {v} with lambda {data['lambda']}, Tao_RP {data['Tao_RP']} and Tao_SW {data['Tao_SW']}")

    for node in G.nodes():
        print(f"Node {node} connected to: {list(G.neighbors(node))}")

    # Create paths from root to all nodes
    # Assuming node 1 is the root (substation)
    root = 1  # Substation node
    paths = {node: nx.shortest_path(G, root, node) for node in G.nodes if node != root}
    print("Shortest paths from root to other nodes:")
    print("Bruto")
    print(paths)
    print("estilizado")
    for node, path in paths.items():
        print(f"Path to node {node}: {path}")
        print("path[:-1]")
        print(path[:-1])
        print("path[1:]")
        print(path[1:])

    
    # Calculate expected number of interruptions per year for each node
    # N_RP
    N_RP = {}
    # Sum the lambda values along the paths
    #  G[u][v]['lambda'] gives the expected number of interruptions per year for edge (u, v)
    #  zip(path[:-1], path[1:]) pairs each node with its successor in the path
    print("Calculating expected number of interruptions per year (N_RP) for each node:")
    for node, path in paths.items():
        N_RP.update({node: sum(G[u][v]['lambda'] for u, v in zip(path[:-1], path[1:]))})
    
    for node, n_rp in N_RP.items():
        print(f"Expected interruptions per year at node {node}: {n_rp:.2f}")

    # Identify wich path belongs to each circuit breaker
    list_of_circuit_breakers = G.neighbors(root)
    list_of_circuit_breakers_by_node = {}
    for node, path in paths.items():
        print(f"Searching in Node {node} with path: {path}")
        list_of_circuit_breakers = G.neighbors(root)
        for cb in list_of_circuit_breakers:
            print(f"Searching in Circuit Breaker {cb}")
            if cb in path:
                list_of_circuit_breakers_by_node.update({node: cb})
                print(f"Node {node} belongs to Circuit Breaker {cb}")
                break

    print("Printing the circuit breaker each path belongs to")
    print(list_of_circuit_breakers_by_node)
    for node, path in paths.items():
        id_cb = list_of_circuit_breakers_by_node.get(node, None)
        print(f"Node: {node} has this path: {path} and belongs to cb: {id_cb}")

    # Integrating the edges belonging to each circuit breaker
    print("Integrating the edges belonging to each circuit breaker")
    list_of_circuit_breakers = G.neighbors(root)
    list_edges_by_circuit_breaker = {cb: [] for cb in list_of_circuit_breakers}
    for node, path in paths.items():
        id_cb = list_of_circuit_breakers_by_node.get(node, None)
        if id_cb is not None:
            # Add edges to the corresponding circuit breaker
            edges = [(u, v) for u, v in zip(path[:-1], path[1:])]
            list_edges_by_circuit_breaker[id_cb].extend(edges)
    print("Edges by circuit breaker:")
    for cb, edges in list_edges_by_circuit_breaker.items():
        print(f"Circuit Breaker {cb}: {edges}")


    # Extracting the exclusive edges belonging to each circuit breaker using the paths
    print("Extracting the exclusive edeges belonging to each circuit breaker")
    list_exclusive_edges_by_cb = {cb: [] for cb in list_of_circuit_breakers}
    for cb, edges in list_edges_by_circuit_breaker.items():
        # Convert edges to a set for uniqueness
        unique_edges = []
        [
            unique_edges.append(edge) for edge in edges if edge not in unique_edges
        ]
        list_exclusive_edges_by_cb[cb] = unique_edges
    print("Exclusive edges by circuit breaker: (Modo 2)")
    for cb, edges in list_exclusive_edges_by_cb.items():
        print(f"Circuit Breaker {cb}: {edges}")

    # Calculate expected number of interruptions per year for each circuit breaker
    print("Calculating expected number of interruptions per year (N_CB) for each circuit breaker:")
    # N_CB will store the total expected interruptions for each circuit breaker
    N_CB = {}
    for cb, edges in list_exclusive_edges_by_cb.items():
        N_CB[cb] = 0
        # Iterate through each edge and sum the lambda values for edges that include circuit breakers
        for u, v in edges:
            if G.has_edge(u, v):
                N_CB[cb] += G[u][v]['lambda']
    for cb, n_cb in N_CB.items():
        print(f"Total expected interruptions per year at circuit breaker {cb}: {n_cb:.2f}")
    
    # Calulate expected number of interruptions per year for each node caused by switching operations
    print("Calculating expected number of interruptions per year (N_SW) for each node caused by switching operations:")
    N_SW = {}
    for node, path in paths.items():
        # First identify the circuit breaker for the node
        id_cb = list_of_circuit_breakers_by_node.get(node, None)
        if id_cb is not None:
            N_SW[node] = N_CB[id_cb] - N_RP[node]

    for node, n_sw in N_SW.items():
        print(f"Expected interruptions per year at node {node} caused by switching operations: {n_sw:.2f}")

    # Visualize the radial distribution network
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    
    plt.title("Radial Distribution Network Topology")
    plt.show()

    # Initialize analyzer
    #analyzer = DistributionTopologyAnalyzer(G, root_node=0)
    
    # Analyze topology
    #results = analyzer.analyze_topology()
    
    # Visualize results for a specific node
    #analyzer.visualize_results(results, load_node=4)