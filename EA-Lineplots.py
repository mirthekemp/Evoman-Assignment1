################################
# Plots                        #
# Author: Group 65             #
#                              #
################################

# # Import framwork and other libs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

if not os.path.exists('EA_LINEPLOTS'):
    os.makedirs('EA_LINEPLOTS')

enemy = 7
# Create a list with all of the .csv files you want to plot
results_list = [f'EA2_enemy{enemy}/EA2_enemy{enemy}.csv']#, f'EA2_enemy{enemy}/EA2_enemy{enemy}.csv']

for result in results_list:

    df = pd.read_csv(result)

    # Grouping the DataFrame by 'Gen' (generation) and aggregating the mean and std
    grouped = df.groupby('Gen').agg({'Mean': ['mean', 'std'],
                                    'Best fit': ['mean', 'std']})

    # Flatten the MultiIndex columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

    # Reset index for seaborn plot
    grouped = grouped.reset_index()

    # Set style for seaborn plot
    sns.set_style("whitegrid")

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot mean fitness with std deviation as a shaded region
    sns.lineplot(data=grouped, x='Gen', y='Mean_mean', label='Average Mean Fitness', color='blue')
    plt.fill_between(grouped['Gen'], 
                    grouped['Mean_mean'] - grouped['Mean_std'], 
                    grouped['Mean_mean'] + grouped['Mean_std'], 
                    color='blue', alpha=0.2)

    # Plot max fitness with std deviation as a shaded region
    sns.lineplot(data=grouped, x='Gen', y='Best fit_mean', label='Average Max Fitness', color='red')
    plt.fill_between(grouped['Gen'], 
                    grouped['Best fit_mean'] - grouped['Best fit_std'], 
                    grouped['Best fit_mean'] + grouped['Best fit_std'], 
                    color='red', alpha=0.2)

    # Set labels and title
    plt.xlabel('Generations')
    plt.ylabel('Fitness Score')
    plt.title(f'Fitness over Generations - EA2 against enemy {enemy}')

    # Add legend
    plt.legend()

    #  create filename
    name = result.replace('.csv', '').split("/")[-1]
    file_name = name + '.png'
    print(file_name)
    
    plt.savefig(f'EA_LINEPLOTS\{file_name}')

    # Show the plot
    plt.show()