import pandas as pd
import matplotlib.pyplot as plt
import ast
import seaborn as sns 

# Read the CSV file into a DataFrame
df = pd.read_csv('EA_BOXPLOTS/data.csv', header=None, names=['Index', 'Data'])
# Parse the string data into lists of numbers using ast.literal_eval
df['Data'] = df['Data'].apply(ast.literal_eval)
# Create a boxplot from the parsed data
custom_palette = sns.color_palette(['red','blue'])
custom_cmap = sns.color_palette(custom_palette)

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(df['Data'], palette=custom_cmap, boxprops=dict(alpha=.6))
ax.set_title(f'Gain Distribution per Algorithm and Enemy', fontsize=16)
ax.set_ylabel('Gain', fontsize=12)
ax.set_xlabel('EAs', fontsize=12, labelpad=27)

ax.grid(axis='y', linestyle='--', alpha=0.7)
original_labels = ['EA1-UM', 'EA2-SAM', 'EA1-UM', 'EA2-SAM', 'EA1-UM', 'EA2-SAM']
ax.set_xticklabels(original_labels, y=-0.015)

grouped_labels  = ['Enemy 1', 'Enemy 6', 'Enemy 7']
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
grouped_x_positions = [0.5, 2.5, 4.5]
ax2.set_xticks(grouped_x_positions)
ax2.set_xticklabels(grouped_labels, y=-0.035, va='center', fontsize=10)
plt.show()











