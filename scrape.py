import pandas as pd
import numpy as np
import twint
import csv

dataset_df = pd.read_csv("data-set-cresci-2018.tsv", sep="\t")
print(dataset_df)

c = twint.Config()
c.Username = "1234a"
c.Followers = True
c.Pandas = True
c.Hide_output = True

twint.run.Followers(c)
df = twint.storage.panda.Follow_df
followers = df["followers"].tolist()[0]

twint.run.Following(c)
df = twint.storage.panda.Follow_df
following = df["following"].tolist()[0]

print(followers)
print(following)
