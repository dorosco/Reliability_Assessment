from pulp import *

# Elementary features:

lp = LpProblem("Bakery_Production", LpMaximize)

# Define Variables
x1 = LpVariable(name = "Bowdoin_log", lowBound=0, cat='Integer') # Integer variable
x2 = LpVariable(name = "Chocolate_cake", lowBound=0, cat='Integer') # Integer variable

# Add the objective function
lp += 10 * x1 + 5 * x2

# Add constraints
lp += 5 * x1 + x2 <= 90, "oven_constraint"
lp += x1 + 10 * x2 <= 300, "food_processor_constraint"
lp += 4 * x1 + 6 * x2 <= 125, "boiler_constraint"
print(lp.constraints) # Print constraints to check them

# Solve the problem
status = lp.solve(PULP_CBC_CMD(msg=0)) # 0: no output, 1: output, 2: verbose
print("Status:", status) #1:optimal, 2:not solved, 3:infeasible, 4:unbounded, 5:undefined
print("Status (Alt):", LpStatus[status])

print("Objective value:", value(lp.objective)) #Optimal value of the objective function
print("Bowdoin log:", x1.varValue) #Optimal value of x1
print("Chocolate cake:", x2.varValue) #Optimal value of x2

#Print solution
print("Solution (from youtube):")
for var in lp.variables():
    print(var.name, "=", var.varValue) #Optimal value of each variable
print("Objective value:", value(lp.objective)) #Optimal value of the objective function


