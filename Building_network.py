import numpy as np
import networkx as nx
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value

class DistributionReliabilityAssessment:
    def __init__(self):
        self.network = nx.Graph()
        self.customers_per_node = {}
        self.branch_failure_rates = {}
        self.branch_repair_times = {}
        
    def add_branch(self, from_node, to_node, length, failure_rate_per_km, repair_time_hours, customers_downstream):
        """
        Add a branch to the network with reliability parameters
        :param from_node: starting node
        :param to_node: ending node
        :param length: branch length in km
        :param failure_rate_per_km: failures per km per year
        :param repair_time_hours: time to repair in hours
        :param customers_downstream: number of customers downstream of this branch
        """
        total_failure_rate = length * failure_rate_per_km
        self.network.add_edge(from_node, to_node, length=length)
        self.branch_failure_rates[(from_node, to_node)] = total_failure_rate
        self.branch_repair_times[(from_node, to_node)] = repair_time_hours
        self.customers_per_node[to_node] = customers_downstream
        
    def calculate_saidi_saifi(self):
        """
        Calculate SAIDI and SAIFI using linear programming approach
        Returns: SAIDI (hours/year), SAIFI (interruptions/year)
        """
        # Total number of customers
        total_customers = sum(self.customers_per_node.values())
        
        # Initialize SAIFI and SAIDI components
        saifi = 0
        saidi = 0
        
        # For each branch, determine which customers are affected when it fails
        for branch in self.branch_failure_rates:
            from_node, to_node = branch
            
            # Create a copy of the network without this branch
            temp_network = self.network.copy()
            temp_network.remove_edge(from_node, to_node)
            
            # Find all nodes that become disconnected from the source (assuming node 0 is source)
            try:
                connected_nodes = nx.node_connected_component(temp_network, 0)
            except:
                # If source is isolated, all nodes are disconnected
                connected_nodes = set()
            
            # Determine affected customers (not in connected component)
            affected_customers = 0
            for node in self.customers_per_node:
                if node not in connected_nodes:
                    affected_customers += self.customers_per_node[node]
            
            # Add contribution to SAIFI and SAIDI
            failure_rate = self.branch_failure_rates[branch]
            repair_time = self.branch_repair_times[branch]
            
            saifi += affected_customers * failure_rate
            saidi += affected_customers * failure_rate * repair_time
        
        # Normalize by total customers
        SAIFI = saifi / total_customers
        SAIDI = saidi / total_customers
        
        return SAIDI, SAIFI
    
    def optimize_reliability(self, budget, cost_per_branch):
        """
        Optimize reliability improvement within budget using linear programming
        :param budget: total available budget
        :param cost_per_branch: dictionary of costs for improving each branch
        Returns: optimal improvement plan
        """
        prob = LpProblem("Reliability_Optimization", LpMinimize)
        
        # Decision variables: whether to improve each branch (binary)
        branch_vars = LpVariable.dicts("ImproveBranch", 
                                      self.branch_failure_rates.keys(),
                                      0, 1, cat='Binary')
        
        # Objective: minimize SAIDI (could also do SAIFI or weighted combination)
        saidi_expr = 0
        for branch in self.branch_failure_rates:
            from_node, to_node = branch
            
            # Create a copy of the network without this branch
            temp_network = self.network.copy()
            temp_network.remove_edge(from_node, to_node)
            
            # Find affected customers
            try:
                connected_nodes = nx.node_connected_component(temp_network, 0)
            except:
                connected_nodes = set()
            
            affected_customers = 0
            for node in self.customers_per_node:
                if node not in connected_nodes:
                    affected_customers += self.customers_per_node[node]
            
            # Original and improved failure rates
            original_failure = self.branch_failure_rates[branch]
            improved_failure = original_failure * 0.2  # Assume improvement reduces failure rate by 80%
            
            # Contribution to SAIDI depends on whether we improve the branch
            saidi_expr += affected_customers * (
                (1 - branch_vars[branch]) * original_failure * self.branch_repair_times[branch] +
                branch_vars[branch] * improved_failure * self.branch_repair_times[branch]
            )
        
        total_customers = sum(self.customers_per_node.values())
        prob += saidi_expr / total_customers  # Minimize SAIDI
        
        # Budget constraint
        prob += lpSum(cost_per_branch[branch] * branch_vars[branch] 
                      for branch in self.branch_failure_rates) <= budget
        
        # Solve the problem
        prob.solve()
        
        # Return improvement plan
        improvement_plan = {
            branch: value(branch_vars[branch]) 
            for branch in self.branch_failure_rates
        }
        
        return improvement_plan

# Example usage
if __name__ == "__main__":
    # Create a simple radial distribution network
    dra = DistributionReliabilityAssessment()
    
    # Add branches (from, to, length_km, failure_rate_per_km, repair_time_hrs, downstream_customers)
    dra.add_branch(0, 1, 1.5, 0.15, 4, 50)  # Feeder from substation
    dra.add_branch(1, 2, 0.8, 0.15, 4, 30)  # Lateral 1
    dra.add_branch(1, 3, 1.2, 0.15, 4, 20)  # Lateral 2
    dra.add_branch(2, 4, 0.5, 0.15, 4, 10)  # Sublateral 1
    dra.add_branch(2, 5, 0.7, 0.15, 4, 20)  # Sublateral 2
    
    # Calculate reliability indices
    SAIDI, SAIFI = dra.calculate_saidi_saifi()
    print(f"Base case SAIDI: {SAIDI:.2f} hours/year")
    print(f"Base case SAIFI: {SAIFI:.2f} interruptions/year")
    
    # Optimize reliability with budget constraint
    cost_per_branch = {
        (0, 1): 50000,
        (1, 2): 30000,
        (1, 3): 35000,
        (2, 4): 20000,
        (2, 5): 25000
    }
    budget = 80000
    plan = dra.optimize_reliability(budget, cost_per_branch)
    
    print("\nOptimal improvement plan:")
    for branch, improve in plan.items():
        if improve > 0.5:
            print(f"Improve branch {branch} (cost: {cost_per_branch[branch]})")