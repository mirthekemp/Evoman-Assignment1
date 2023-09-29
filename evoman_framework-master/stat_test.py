################################
# Plots                        #
# Author: Group 65             #
#                              #
################################

# # Import framwork and other libs
import pandas as pd
from scipy import stats

def create_group(results_file, generation=29):
    '''
    This function creates a list of the mean fitness values at a specific generation number.
    '''
    df = pd.read_csv(results_file)

    # To select values from the column 'Mean' where 'Gen' == generation:
    group = list(df[df['Gen'] == generation]['Mean'])

    return group

def t_test(group_a, group_b):
    ''''
    Performs an independent two-sample t-test between two groups
    '''
    # Perform independent two-sample t-test
    t_statistic, p_value = stats.ttest_ind(group_a, group_b)

    # Print the results
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")

    # Compare p-value to the chosen significance level (α)
    alpha = 0.05
    if p_value < alpha:
        print(f"Reject null hypothesis: There is a significant difference between the groups.")
    else:
        print(f"Fail to reject null hypothesis: There is no significant difference between the groups.")
    return

# Creata a list of mean fitness values at generation 29 for each group
ea2_enemy1 = create_group('EA2_enemy1_param7\EA2_enemy1_param7.csv', 29)
ea2_enemy6 = create_group('EA2_enemy6_param7\EA2_enemy6_param7.csv', 29)

t_test(ea2_enemy1, ea2_enemy6)
