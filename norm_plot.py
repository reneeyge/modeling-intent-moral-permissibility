import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model_data = [0.277908761, 0.275800102, 0.280426084, 0.9399871, 0.926234784, 0.924426234]
human_data = [0.2184, 0.6279, 0.3732, 0.9454, 0.8908, 0.9545]

# Set position of bar on X axis
barWidth = 0.1
r1 = np.arange(len(model_data))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
bands =  ["5vB", "2vB", "1vB", "Bv1", "Bv2", "Bv5"]

# Make the plot
plt.bar(r1, model_data, width=barWidth, edgecolor='white', label='Model')
plt.bar(r2, human_data, width=barWidth, edgecolor='white', label='Human Data')

# Add xticks on the middle of the group bars
plt.ylabel('Norm = Only loved ones are valued', fontweight='bold')
plt.xlabel('Scenario', fontweight='bold')
plt.title("Likelihood of agent believing in norm given the action of pulling the switch")
plt.xticks([r + 1/3*barWidth for r in range(len(model_data))], bands, rotation = 30)
plt.legend()
sns.despine()

plt.show()
plt.savefig('norm_plot')