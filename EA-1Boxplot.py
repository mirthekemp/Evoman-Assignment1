import pandas as pd
import matplotlib.pyplot as plt
import ast
import seaborn as sns 

# Read the CSV file into a DataFrame
df = pd.read_csv('EA_BOXPLOTS/data3.csv', header=None, names=['Index', 'Data'])
# Parse the string data into lists of numbers using ast.literal_eval
df['Data'] = df['Data'].apply(ast.literal_eval)
# Create a boxplot from the parsed data
custom_palette = sns.color_palette(['dimgray', 'lightgray' ])
custom_cmap = sns.color_palette(custom_palette)

fig, ax = plt.subplots(figsize=(9, 7))
sns.boxplot(df['Data'], palette=custom_cmap, boxprops=dict(alpha=0.45))
ax.set_title(f'Gain Distribution per Enemy and Algorithm', fontsize=16, fontweight='bold')
ax.set_ylabel('Gain', fontsize=14, fontweight='bold')
ax.set_xlabel('EAs', fontsize=14, fontweight='bold', labelpad=10)

ax.grid(axis='y', linestyle='--', alpha=0.5)
original_labels = ['NUM', 'SAM', 'NUM', 'SAM', 'NUM', 'SAM']
ax.set_xticklabels(original_labels, y=-0.03)
for line in ax.lines:
    if len(line.get_xdata()) == 2:  # Median line has 2 x-values
        line.set(lw=1)  # Set the line width to make it bold
grouped_labels  = ['Enemy 1', 'Enemy 6', 'Enemy 7']
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
grouped_x_positions = [0.5, 2.5, 4.5]
ax2.set_xticks(grouped_x_positions)
ax2.set_xticklabels(grouped_labels, y=-0.045, va='center')
plt.savefig(f'EA_BOXPLOTS\Boxplots_EAvEA2_all_enemies_3')
plt.show()












