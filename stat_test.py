################################
# Plots                        #
# Author: Group 65             #
#                              #
################################

# # Import framwork and other libs
import os
import pandas as pd
from scipy import stats
import numpy as np

def create_group(results_file, generation=29):
    '''
    This function creates a list of the mean fitness values at a specific generation number.
    '''
    df = pd.read_csv(results_file)

    # To select values from the column 'Mean' where 'Gen' == generation:
    group = list(df[df['Gen'] == generation]['Mean'])

    return group

def descriptive_stats(group, group_name):
    '''
    This function calculates and prints the mean and standard deviation of a group.
    '''
    mean = sum(group) / len(group)
    std_dev = np.std(group)

    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std_dev}")

    print(f"{group_name} Mean: {mean}")
    print(f"{group_name} Standard Deviation: {std_dev}")
    print()

def t_test(group_a, group_b):
    ''''
    Performs an independent two-sample t-test between two groups
    '''
    # Perform independent two-sample t-test
    t_statistic, p_value = stats.ttest_ind(group_a, group_b)

    # Calculate degrees of freedom
    dof = len(group_a) + len(group_b) - 2

    # Print the results
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")
    print(f"Degrees of freedom: {dof}")
    


    # Compare p-value to the chosen significance level (α)
    alpha = 0.05
    if p_value < alpha:
        print(f"Reject null hypothesis: There is a significant difference between the groups.")
    else:
        print(f"Fail to reject null hypothesis: There is no significant difference between the groups.")
    return


for enemy_nr in [1, 6, 7]:

    print(f"ENEMY {enemy_nr}: EA1 vs EA2\n")

    # Perform a t-test between the two groups at generation 4, 9 and 29
    for gen in [29]: # [4, 9, 29]:

        csv_EA1 = f'EA1_NUM2_enemy{enemy_nr}/EA1_NUM2_enemy{enemy_nr}.csv' #oud: csv_EA1 = f'EA1_enemy{enemy_nr}/EA1_enemy{enemy_nr}.csv'
        csv_EA2 = f'EA2_enemy{enemy_nr}/EA2_enemy{enemy_nr}.csv'

        data_EA1 = create_group(csv_EA1, gen)
        data_EA2 = create_group(csv_EA2, gen)

        print("Generation:", gen)
        descriptive_stats(data_EA1, "EA1")
        descriptive_stats(data_EA2, "EA2")
        t_test(data_EA1, data_EA2)
        print()
