import pulp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class DistributionTopologyAnalyzer:
    def __init__(self, network_graph, root_node=0):
        """
        Initialize the topology analyzer with a radial network graph
        
        Args:
            network_graph: NetworkX graph representing the radial distribution network
            root_node: The root/substation node (default: 0)
        """
        self.G = network_graph
        self.root = root_node
        self.nodes = list(network_graph.nodes())
        self.edges = list(network_graph.edges())
        
        # Add reverse edges for modeling purposes (power can flow both ways in model)
        self.directed_edges = self.edges + [(v, u) for (u, v) in self.edges]
        
        # Precompute paths from root to all nodes
        self.paths = {node: nx.shortest_path(self.G, self.root, node) for node in self.nodes if node != self.root}
        
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
        # Power flow on each edge (continuous, can be positive or negative)
        P = pulp.LpVariable.dicts("PowerFlow", self.directed_edges, cat='Continuous')
        
        # Voltage magnitude at each node (relative to root)
        V = pulp.LpVariable.dicts("Voltage", self.nodes, lowBound=0, upBound=1.1, cat='Continuous')
        
        # Dummy objective - we're just looking for feasible solutions
        prob += 0, "Dummy objective"
        
        # Constraints
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
        (0, 1, {'resistance': 0.02, 'reactance': 0.02}),  # Substation to first feeder
        (1, 2, {'resistance': 0.03, 'reactance': 0.03}),  # Main feeder
        (2, 3, {'resistance': 0.01, 'reactance': 0.01}),  # Lateral
        (2, 4, {'resistance': 0.01, 'reactance': 0.01}),  # Lateral
        (1, 5, {'resistance': 0.04, 'reactance': 0.04}),  # Another main branch
        (5, 6, {'resistance': 0.02, 'reactance': 0.02}),  # Lateral
        (5, 7, {'resistance': 0.02, 'reactance': 0.02})   # Lateral
    ]
    G.add_edges_from(edges)
    
    # Initialize analyzer
    analyzer = DistributionTopologyAnalyzer(G, root_node=0)
    
    # Analyze topology
    results = analyzer.analyze_topology()
    
    # Visualize results for a specific node
    analyzer.visualize_results(results, load_node=4)