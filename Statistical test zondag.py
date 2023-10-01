# Statistical test

# # Import framwork and other libs
import os
import pandas as pd
from scipy import stats
import numpy as np

def create_group(df, generation=29):
    '''
    This function creates a list of the mean fitness values at a specific generation number.
    '''

    # To select values from the column 'Mean' where 'Gen' == generation:
    group = list(df[df['Gen'] == generation]['Mean'])
    print(group)

    return group


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

enemy_nr = 7
csv_NUM = pd.read_csv(f'EA1_NUM2_enemy{enemy_nr}\EA1_NUM2_enemy{enemy_nr}.csv')
csv_SAM = pd.read_csv(f'EA2_enemy{enemy_nr}\EA2_enemy{enemy_nr}.csv')

group_NUM = create_group(csv_NUM)
group_SAM = create_group(csv_SAM)
t_test(group_NUM, group_SAM)