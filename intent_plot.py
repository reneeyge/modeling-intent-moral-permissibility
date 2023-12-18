
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model_data = [0.033821777, 0.255182789, 0.082900495, 
              0.277950774, 0.162973162, 0.30249444, 
              0.067307358, 0.243347983, 0.076269175, 
              0.444233919, 0.081700562]
human_data = [0, 0.182, 0, 0.182, 0.727, 0.909, 0.364, 0.818, 0.455, 0.727, 0.455]

# Set position of bar on X axis
barWidth = 0.1
r1 = np.arange(len(model_data))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
bands =  ["5v1", "5vB", "2v1", "2vB", "1v1", "1vB", "Bv1", "1v2", "Bv2", "1v5", "Bv5"]

# Make the plot
plt.bar(r1, model_data, width=barWidth, edgecolor='white', label='Model')
plt.bar(r2, human_data, width=barWidth, edgecolor='white', label='Human Data')

# Add xticks on the middle of the group bars
plt.ylabel('Intention of killing people on side track', fontweight='bold')
plt.xlabel('Track Configuration', fontweight='bold')
plt.title("Intention Inference")
plt.xticks([r + 1/3*barWidth for r in range(len(model_data))], bands, rotation = 30)
plt.legend()
sns.despine()

plt.show()
plt.savefig('intention')