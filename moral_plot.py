
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model_data = [0.738835066386733, 0.5085295445985663, 0.7415065282234399, 
              0.5159192780837973, 0.678367338325259, 0.4965078727255734, 
              0.7688094663442087, 0.5931780022315986, 0.7766718442305506, 
              0.36734560606074773, 0.8110870168149635]
human_data = [1, 0.909, 0.909, 0.818, 0.091, 0.091, 0.818, 0, 0.455, 0, 0.364]

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
plt.ylabel('Morally permissible', fontweight='bold')
plt.xlabel('Track configuration', fontweight='bold')
plt.title("Moral Permissibility")
plt.xticks([r + 1/3*barWidth for r in range(len(model_data))], bands, rotation = 30)
plt.legend()
sns.despine()

plt.show()
plt.savefig('moral_plot')