import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

df = pd.read_csv('../../data/train_graph.csv',encoding='utf8',engine='python',chunksize=None)

human = df[df.label == 0]
bot   = df[df.label == 1]

data_h = human['favourites_count']
data_b = bot['favourites_count']
weights_h = np.ones_like(data_h)/float(len(data_h))
weights_b = np.ones_like(data_b)/float(len(data_b))

plt.figure()
plt.hist([data_h, data_b],weights=[weights_h, weights_b], bins=50,density=False,color=["blue","red"],cumulative=False,label=["Human","Bot"],histtype='bar',range=(0,5000))
plt.legend()
plt.ylabel("Frequency")
plt.xlabel("Number of favourites")
#plt.show()
plt.savefig("favourites_count.pdf", dpi=300)
