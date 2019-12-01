import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)

print("Loading dataset...")
df = pd.read_csv('../data/train_baseline.csv', encoding='utf8', engine='python', chunksize=None)

features = list(df.columns)
features.remove('label')

human = df[df['label'] == 0]
bot = df[df['label'] == 1]

human = human[features]
bot = bot[features]

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
cdict = {'0': 'blue', 1: 'red'}

ax.set_xlim3d(0,np.log10(max(df['favourites_count'])/10))
ax.set_ylim3d(0,np.log10(max(df['followers'])/10))
ax.set_zlim3d(0,6)

human_xs = human['favourites_count']
human_ys = human['followers']
human_zs = human['statuses_count']

bot_xs = bot['favourites_count']
bot_ys = bot['followers']
bot_zs = bot['statuses_count']

ax.scatter(np.log10(human_xs), np.log10(human_ys), np.log10(human_zs), s=15, alpha=.95, edgecolors="k", linewidths=0.5, color="dodgerblue", label="Human")
ax.scatter(np.log10(bot_xs), np.log10(bot_ys), np.log10(bot_zs), s=15, alpha=.95, edgecolors="k", linewidths=0.5, color="palevioletred", label="Bots")

ax.set_xlabel('Favorites')
ax.set_ylabel('Following')
ax.set_zlabel('Status')

plt.legend()

plt.show()
fig.savefig('../paper/FIG/3dvis_base.pdf')
