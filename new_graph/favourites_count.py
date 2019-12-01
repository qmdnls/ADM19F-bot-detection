import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

df = pd.read_csv('../data/train_graph.csv',encoding='utf8',engine='python',chunksize=None)

human = df[df.label == 0]
bot   = df[df.label == 1]

plt.figure()
plt.hist([human['statuses_count'],bot['statuses_count']],bins=50,density=False,color=["blue","red"],cumulative=False,label=["Human","Bot"],histtype='bar',range=(0,5000))
#plt.hist(bot['status_successors'],bins=250,density=False,range=(0,40000),color="red",cumulative=False,label="bots",histtype='bar') 
plt.legend()
plt.ylabel("Frequency")
plt.xlabel("Number of status")
#plt.show()
plt.savefig("status_count.pdf", dpi=300)
