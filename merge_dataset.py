import pandas as pd


af = pd.read_csv("data2.csv", encoding='utf8', engine='python')
bf = pd.read_csv("progress_final.csv", encoding='utf8', engine='python')
af['user_id'] = af['user_id'].astype(int)
print(af.head)
print(bf.head)
df = pd.merge(af, bf, on='user_id')
print(df.head)

df.to_csv("dataset.csv", index=None, header=True)
