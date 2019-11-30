import pandas as pd

#af = pd.read_csv("data2.csv", encoding='utf8', engine='python')
#bf = pd.read_csv("progress_final.csv", encoding='utf8', engine='python')
#af['user_id'] = af['user_id'].astype(int)
#df = pd.merge(af, bf, on='user_id')
#print(df.head)

af = pd.read_csv("dataset.csv", encoding="utf8", engine="python")
bf = pd.read_csv("corrected.csv", encoding="utf8", engine="python")

df = pd.concat([af, bf])

df.to_csv("users.csv", index=None, header=True)
