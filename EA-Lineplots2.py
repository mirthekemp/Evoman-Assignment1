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
alg = 'UM'
# Create a list with all of the .csv files you want to plot
result = f'EA1_enemy{enemy}/EA1_enemy{enemy}.csv'

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
plt.figure(figsize=(9, 7))

# Plot mean fitness with std deviation as a shaded region
sns.lineplot(data=grouped, x='Gen', y='Mean_mean', label='Average Mean Fitness', color='blue')
plt.fill_between(grouped['Gen'], 
                grouped['Mean_mean'] - grouped['Mean_std'], 
                grouped['Mean_mean'] + grouped['Mean_std'], 
                color='blue', alpha=0.2)

sns.scatterplot(data=grouped, x='Gen', y='Mean_mean', color='blue', s=50) # 's' controls the size of the dots

# Plot max fitness with std deviation as a shaded region
sns.lineplot(data=grouped, x='Gen', y='Best fit_mean', label='Average Max Fitness', color='red')
plt.fill_between(grouped['Gen'], 
                grouped['Best fit_mean'] - grouped['Best fit_std'], 
                grouped['Best fit_mean'] + grouped['Best fit_std'], 
                color='red', alpha=0.2)

sns.scatterplot(data=grouped, x='Gen', y='Best fit_mean', color='red', s=50) # 's' controls the size of the dots


# Set labels and title
plt.xlim(0, 29)
plt.ylim(-5, 105)
plt.xticks(range(0, 30, 2), fontsize=16)
plt.yticks(range(0, 105, 10), fontsize=16)
plt.xlabel('Generations', fontsize=18, fontweight='bold')
plt.ylabel('Fitness Score', fontsize=18, fontweight='bold')
plt.title(f'{alg} - Enemy {enemy}', fontsize=20, fontweight='bold')

# Add legend
plt.legend(loc='lower right', fontsize=18)

#  create filename
name = result.replace('.csv', '').split("/")[-1]
file_name = name + '.png'
print(file_name)

plt.savefig(f'EA_LINEPLOTS/2{file_name}')

# Show the plot
plt.show()