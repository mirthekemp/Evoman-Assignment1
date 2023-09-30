import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import EA1_FLOOR 
import ast  # Import the ast module for literal_eval
from statistics import mean
import csv


"""
Boxplot
    - Comparing 10 final solutions (max fitness after 30 generations)
    - Run every solution 5 times
    - Should return the same result 5 times (small odds for different outcome)
    - Take the average gain of these 5 results
    - Create a boxplot with the 10 averages
    - Y-axis: gain = p_health - e_health
    - Statistical test to prove that they are significantly different
"""
if not os.path.exists('EA_BOXPLOTS'):
    os.makedirs('EA_BOXPLOTS')

enemy = 7
results_list = [f'EA1_NUM2_enemy{enemy}/EA1_NUM2_enemy{enemy}.csv', f'EA2_enemy{enemy}/EA2_enemy{enemy}.csv']
_10runs_2enemies = []

for result in results_list:
    df = pd.read_csv(result)
    selected_rows = df[df['Gen'] == EA1_FLOOR.n_generations - 1]  # Last gen in this case, so this is the best solution
    print(selected_rows)
    averages_10_runs = []
    for solution in selected_rows['BEST SOL']:
        list_sol = ast.literal_eval(solution)
        array_solutions = np.array(list_sol).reshape(1, 265)
        repeat5 = []
        for _ in range(5):
            gain = round(float(EA1_FLOOR.evaluate(array_solutions)[:, 1][0]), 6)
            repeat5.append(gain)  
        #print(repeat5) # if you print this, you will see that it is five times the same value
        avg_5repeats = mean(repeat5)
        averages_10_runs.append(avg_5repeats)
    _10runs_2enemies.append(averages_10_runs)
print(_10runs_2enemies)

for sublist in enumerate(_10runs_2enemies):
    with open(f'EA_BOXPLOTS/data3.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(sublist)



