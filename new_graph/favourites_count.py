import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

df = pd.read_csv('train_graph.csv',encoding='utf8',engine='python',chunksize=None)

human = df[df.label == 0]
bot   = df[df.label == 1]

plt.figure()
plt.hist([human['status_successors'],bot['status_successors']],bins=50,density=False,range=(0,4000),color=["blue","red"],cumulative=False,label=["Human","Bot"],histtype='bar') 
#plt.hist(bot['status_successors'],bins=250,density=False,range=(0,40000),color="red",cumulative=False,label="bots",histtype='bar') 
plt.legend()
plt.ylabel("frequency")
plt.xlabel("status_successors")
#plt.show()
plt.savefig("deneme.png"); 


