from pulp import *

# Advanced features:

lp = LpProblem("Bakery_Production", LpMaximize)

# Define a dcitionary of variables keyed by "indices"
var_keys = [1, 2]
x = LpVariable.dicts("Bakery_item", var_keys, lowBound=0, cat='Integer') # Integer variable
print(x) # Print the dictionary of variables

# Add the objective function
lp += 10 * x[1] + 5 * x[2]

# Add constraints
#lp += 5 * x[1] + x[2] <= 90, "oven_constraint"
#lp += lpSum( [5*x[1], x[2]] ) <= 90, "oven_constraint" # Using lpSum for constraints
lp += x[1] + 10 * x[2] <= 300, "food_processor_constraint"
lp += 4 * x[1] + 6 * x[2] <= 125, "boiler_constraint"

#Rewrite the first constraint using lpSum and zip
coeff = [5, 1] # may come from a data file
coeff_dict = dict(zip(var_keys, coeff))
lp += lpSum( [coeff_dict[i] * x[i] for i in var_keys] ) <= 90, "oven_constraint" # Using lpSum for constraints

print(lp.constraints) # Print constraints to check them


# Solve the problem
status = lp.solve(PULP_CBC_CMD(msg=0)) # 0: no output, 1: output, 2: verbose
print("Status:", status) #1:optimal, 2:not solved, 3:infeasible, 4:unbounded, 5:undefined
print("Status (Alt):", LpStatus[status])

print("Objective value:", value(lp.objective)) #Optimal value of the objective function
print("Bowdoin log:", x[1].varValue) #Optimal value of x1
print("Chocolate cake:", x[2].varValue) #Optimal value of x2

#Print solution
print("Solution (from youtube):")
for var in lp.variables():
    print(var.name, "=", var.varValue) #Optimal value of each variable
print("Objective value:", value(lp.objective)) #Optimal value of the objective function
